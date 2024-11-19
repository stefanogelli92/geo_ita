import difflib
import os
import logging
import ssl
from datetime import datetime
from typing import Dict, Union, List
import unidecode
import requests

from valdec.decorators import validate
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import geopandas as gpd
from geopy.distance import distance
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from googlesearch import search
from googleapiclient.discovery import build
# from sklearn.neighbors import KernelDensity

from bokeh.models import (
    ColumnDataSource, DataTable, TableColumn, HTMLTemplateFormatter, CategoricalColorMapper,
    LabelSet, Label, WheelZoomTool, CustomJS, TabPanel, Tabs, TextInput, HoverTool
)
from bokeh.plotting import save, figure
from bokeh.io import output_file, show
from bokeh.layouts import column, row
import xyzservices.providers as xyz
from pyproj import Proj, transform

from geo_ita.src.definition import *
from geo_ita.src.utils import *
import geo_ita.src.config as cfg
from geo_ita.src._data import (
    get_df, get_df_comuni, get_administrative_changes_df, _clean_denom_text,
    get_double_languages_denomination,
    _get_shape_italia, get_high_resolution_population_density_df, get_highway_shapes, get_highway_exits
)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx

inProj, outProj = Proj(init='epsg:4326'), Proj(init='epsg:3857')


def google_query(query, api_key, cse_id, **kwargs):
    query_service = build("customsearch",
                          "v1",
                          developerKey=api_key
                          )
    query_results = query_service.cse().list(q=query,  # Query
                                             cx=cse_id,  # CSE ID
                                             **kwargs
                                             ).execute()
    return query_results['items']


@validate
def _clean_htmltext(text: str) -> str:
    text = text.lower()
    text = re.sub('[^A-Za-z0-9.]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(' +', ' ', text)
    return text


@validate
def _test_column_in_dataframe(df: pd.DataFrame, column):
    if column not in df.columns:
        raise Exception(f"Column {column} not found in DataFrame.")


class AddGeographicalInfo:
    MATCH_COLUMN = "geo_ita_match_column"
    SUFFIX_DEFAULT = "_geo_ita_suffix_default"
    OUTPUT_COLUMNS = [
        cfg.TAG_COMUNE, cfg.TAG_CODICE_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
        cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE, cfg.TAG_AREA_GEOGRAFICA, cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE
    ]

    @validate
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.where(pd.notnull(df), None)
        self.detail_level = {}

        self.df = None
        self.istat_registry = None
        self.not_match = None

    @validate
    def set_comuni_tag(self, column_name: str):
        _test_column_in_dataframe(self.original_df, column_name)
        code_level = infer_geographical_category(list(self.original_df[column_name].unique()))
        if code_level == CodeLevel.SIGLA:
            raise Exception(f"Found values in {column_name} similar to Province Sigla. "
                            f"Check the column name passed and the values on columns.")
        self.detail_level[GeoLevel.COMUNE] = (column_name, code_level)

    @validate
    def set_province_tag(self, column_name: str):
        _test_column_in_dataframe(self.original_df, column_name)
        code_level = infer_geographical_category(list(self.original_df[column_name].unique()))
        self.detail_level[GeoLevel.PROVINCIA] = (column_name, code_level)

    @validate
    def set_regioni_tag(self, column_name: str):
        _test_column_in_dataframe(self.original_df, column_name)
        code_level = infer_geographical_category(list(self.original_df[column_name].unique()))
        if code_level == CodeLevel.SIGLA:
            raise Exception(f"Found values in {column_name} similar to Province Sigla. "
                            f"Check the column name passed and the values on columns.")
        self.detail_level[GeoLevel.REGIONE] = (column_name, code_level)

    def run_simple_match(self):
        if len(self.detail_level) == 0:
            raise Exception("You need to set al least one between comuni_tag, province_tag or regioni_tag.")

        # Sort detail_level
        self.detail_level = dict(sorted(self.detail_level.items()))
        geo_level = list(self.detail_level.keys())[0]
        geo_code = list(self.detail_level.values())[0][1]

        # Create all combination of geographical data
        geo_columns = [name for name, _ in self.detail_level.values()]
        self.df = self.original_df[geo_columns].copy().drop_duplicates()

        # Create match column
        self.df[self.MATCH_COLUMN] = self.df[geo_columns[0]]

        # The main geographical level need to be not null
        self.df = self.df[self.df[self.MATCH_COLUMN].notnull()]

        # Get ISTAT data
        self.istat_registry = get_df(geo_level)

        geo_tag_anag = _get_tag_anag(geo_code, geo_level)
        self.istat_registry[self.MATCH_COLUMN] = self.istat_registry[geo_tag_anag]

        if geo_code == CodeLevel.SIGLA:
            self._run_sigla_match()
        elif geo_code == CodeLevel.CODE:
            self._run_code_match()
        else:
            self._run_denomination_match()

    def _check_non_match(self, istat_values):
        self.not_match = self.df[(~self.df[self.MATCH_COLUMN].isin(istat_values))]
        return self.not_match[self.MATCH_COLUMN].nunique()

    def _run_sigla_match(self):
        # Sigla Cleaning
        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].str.lower()
        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].str.strip()
        self.istat_registry[self.MATCH_COLUMN] = self.istat_registry[self.MATCH_COLUMN].str.lower()

        istat_values = list(self.istat_registry[self.MATCH_COLUMN].unique())

        n_not_match = self._check_non_match(istat_values)
        n_tot = self.df[self.MATCH_COLUMN].nunique()
        if n_not_match == 0:
            log.info(f"Matching completed, found {n_tot} different sigle.")
        else:
            log.warning(f"Matched {n_tot - n_not_match} over {n_tot}."
                        f" Missing {n_not_match} unique values ({n_not_match / n_tot:.1%}).")

    def _run_code_match(self):
        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].astype(int, errors='ignore')
        self.istat_registry[self.MATCH_COLUMN] = self.istat_registry[self.MATCH_COLUMN].astype(int, errors='ignore')

        istat_values = list(self.istat_registry[self.MATCH_COLUMN].unique())

        n_not_match = self._check_non_match(istat_values)

        n_tot = self.df[self.MATCH_COLUMN].nunique()

        if n_not_match == 0:
            log.info(f"Matching completed, found {n_tot} different codes.")
        else:
            log.warning(f"Matched {n_tot - n_not_match} over {n_tot}."
                        f" Missing {n_not_match} unique values ({n_not_match / n_tot:.1%}).")

    def _run_denomination_match(self):
        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].str.lower()

        geo_level = list(self.detail_level.keys())[0]

        self.df[self.MATCH_COLUMN] = _clean_denom_text(self.df[self.MATCH_COLUMN])
        self.istat_registry[self.MATCH_COLUMN] = _clean_denom_text(self.istat_registry[self.MATCH_COLUMN])

        if geo_level == GeoLevel.COMUNE:
            self._test_if_df_contains_homonym_comuni()

        self._find_any_bilingual_name()

        self._rename_any_english_name()

        if geo_level == GeoLevel.COMUNE:
            self._find_any_variation_from_istat_history()

        istat_values = list(self.istat_registry[self.MATCH_COLUMN].unique())

        _ = self._check_non_match(istat_values)

        self._try_custom_replace_denomination(istat_values)
        n_not_match = self._check_non_match(istat_values)

        n_tot = self.df[self.MATCH_COLUMN].nunique()
        if n_not_match == 0:
            log.info(f"Matching completed, found {n_tot} different names.")
        else:
            log.warning(f"Matched {n_tot - n_not_match} over {n_tot}.")

    @validate
    def get_not_matched_list(self) -> List[str]:
        if self.not_match is None:
            raise Exception("Run simple match before getting list of not matched values.")
        result = list(self.not_match[self.MATCH_COLUMN].unique())
        return result

    def _add_provincia_regione_denomination_to_not_matched(self):
        if GeoLevel.PROVINCIA in self.detail_level:
            detail_column = self.detail_level[GeoLevel.PROVINCIA][0]
            addinfo = AddGeographicalInfo(self.not_match)
            addinfo.set_province_tag(detail_column)
            addinfo.run_simple_match()
            self.not_match = addinfo.get_result(suffix_result_columns=self.SUFFIX_DEFAULT)
        elif GeoLevel.REGIONE in self.detail_level:
            detail_column = self.detail_level[GeoLevel.REGIONE][0]
            addinfo = AddGeographicalInfo(self.not_match)
            addinfo.set_regioni_tag(detail_column)
            addinfo.run_simple_match()
            self.not_match = addinfo.get_result(suffix_result_columns=self.SUFFIX_DEFAULT)

    @validate
    def get_n_not_matched(self) -> int:
        if self.not_match is None:
            raise Exception("Run simple match before getting number of not matched values.")
        result = self.not_match[self.MATCH_COLUMN].nunique()
        return result

    def _test_if_df_contains_homonym_comuni(self):
        comuni_homonym_df = self._calculate_italian_comuni_homonym()
        # Check if the dataset contains homonym comune
        homonym_comuni_list = list(set(comuni_homonym_df[cfg.TAG_COMUNE]))
        if (self.df[self.MATCH_COLUMN].isin([a.lower() for a in homonym_comuni_list])).sum() == 0:
            return

        log.info(f"Found homonym comuni on dataset.")
        if GeoLevel.PROVINCIA in self.detail_level:
            geo_code = self.detail_level[GeoLevel.PROVINCIA][1]
            detail_column = self.detail_level[GeoLevel.PROVINCIA][0]
            registry_column_detail = _get_tag_anag(geo_code, GeoLevel.PROVINCIA)
        elif GeoLevel.REGIONE in self.detail_level:
            geo_code = self.detail_level[GeoLevel.REGIONE][1]
            detail_column = self.detail_level[GeoLevel.REGIONE][0]
            registry_column_detail = _get_tag_anag(geo_code, GeoLevel.REGIONE)
        else:
            log.warning(
                "You can distinguish them only by using another geographic information (ex.: provincia or regione). "
                "If you want to identify the right comune add provincia or regione detail or homonym comuni will be "
                "ignored.")
            return
        log.info(f"The column {detail_column} will be used in order to found the right comune.")
        comuni_homonym_df["key"] = comuni_homonym_df[cfg.TAG_COMUNE] + " " + comuni_homonym_df[registry_column_detail]
        self.df = self._split_comuni_homonym(self.df, detail_column, comuni_homonym_df)
        self.istat_registry = self._split_comuni_homonym(self.istat_registry, registry_column_detail, comuni_homonym_df)

    def _calculate_italian_comuni_homonym(self):
        df = self.istat_registry
        df = df[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]].copy()
        for column in df.columns:
            df[column] = df[column].str.lower()
        df["name_count"] = df.groupby(cfg.TAG_COMUNE)[cfg.TAG_PROVINCIA].transform("count")
        df = df[df["name_count"] > 1]
        df.drop(columns=["name_count"], inplace=True)
        df["new_name"] = df[cfg.TAG_COMUNE] + " " + df[cfg.TAG_SIGLA]
        return df

    @staticmethod
    def _split_comuni_homonym(df, details_columns, comuni_homonym_df):
        denomination_column = AddGeographicalInfo.MATCH_COLUMN

        pos = df[denomination_column].isin(comuni_homonym_df[cfg.TAG_COMUNE].unique())
        df[denomination_column] = df[denomination_column].where(~pos,
                                                                df[denomination_column] + " " + df[details_columns].astype(
                                                                    str).str.lower())
        df[denomination_column] = df[denomination_column].replace(comuni_homonym_df.set_index("key")["new_name"])
        return df

    def _find_any_bilingual_name(self):
        geo_level = list(self.detail_level.keys())[0]
        replace_multilanguage_name = get_double_languages_denomination(geo_level, self.istat_registry)

        for k, v in replace_multilanguage_name.items():
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(k, v)
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(f"{k} {v}", v)
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(f"{v} {k}", v)

    def _rename_any_english_name(self):
        s: pd.Series = self.df[self.MATCH_COLUMN].replace(cfg.rename_english_name)
        match_english_name = s != self.df[self.MATCH_COLUMN]
        if match_english_name.any():
            replaces = self.df[match_english_name][self.MATCH_COLUMN].unique()
            log.info(f"Replaced {len(replaces)} name written in english: {replaces}")
            self.df[self.MATCH_COLUMN] = s

    def _find_any_variation_from_istat_history(self, df_changes=None, i=0):
        if df_changes is None:
            df_changes = get_administrative_changes_df()
            df_changes["new_denominazione_comune"] = df_changes["new_denominazione_comune"].where(
                ~df_changes["new_denominazione_comune"].str.contains(r"/"),
                df_changes["new_denominazione_comune"].str.split(r"/").str[0]
            )
            df_changes[cfg.TAG_COMUNE] = df_changes[cfg.TAG_COMUNE].where(
                ~df_changes[cfg.TAG_COMUNE].str.contains(r"/"),
                df_changes[cfg.TAG_COMUNE].str.split(r"/").apply(lambda x: x + [r"/".join(x), r"/".join(x[::-1])])
            )
            df_changes = df_changes.explode(cfg.TAG_COMUNE)
            df_changes[cfg.TAG_COMUNE] = _clean_denom_text(df_changes[cfg.TAG_COMUNE])
            df_changes["new_denominazione_comune"] = _clean_denom_text(df_changes["new_denominazione_comune"])
            df_changes["data_decorrenza"] = pd.to_datetime(df_changes["data_decorrenza"])
            df_changes.sort_values([cfg.TAG_COMUNE, "data_decorrenza"], ascending=False, inplace=True)
            df_changes = df_changes.groupby(cfg.TAG_COMUNE)["new_denominazione_comune"].last().to_dict()

        s: pd.Series = self.df[self.MATCH_COLUMN].where(
            self.df[self.MATCH_COLUMN].isin(list(self.istat_registry[self.MATCH_COLUMN].unique())),
            self.df[self.MATCH_COLUMN].replace(df_changes)
        )
        if (s != self.df[self.MATCH_COLUMN]).any():
            replaces = s[(s != self.df[self.MATCH_COLUMN])].unique()
            self.df[self.MATCH_COLUMN] = s
            log.info(f"Match {len(replaces)} name that are no longer comuni:\n{replaces}")

            # Check again for possible changes in istat registry for any old comune that need more than one change
            # in order to find the current name of the comune.
            replaces_correct = [name for name in replaces if name in self.istat_registry[self.MATCH_COLUMN].values]
            if len(replaces_correct) > 0:
                self._find_any_variation_from_istat_history(df_changes=df_changes, i=i + 1)

    def _try_custom_replace_denomination(self, istat_values):
        list_den_not_found = self.get_not_matched_list()
        dict_den_anag = {self._custom_replace_denomination(a): a for a in istat_values}
        dict_den_not_found = {a: self._custom_replace_denomination(a) for a in list_den_not_found}
        dict_den_not_found = {k: dict_den_anag[v] for k, v in dict_den_not_found.items() if v in dict_den_anag}
        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(dict_den_not_found)

    @staticmethod
    def _custom_replace_denomination(value):
        for v in cfg.clear_den_replace:
            value = value.replace(v[0], v[1])
        value = " ".join(value.split())
        return value

    @staticmethod
    def _check_if_comune_is_comune(value):
        info = get_geo_info_from_comune(comune=value, flag_find_frazioni=False)
        if info is None:
            return None
        else:
            return info[cfg.TAG_COMUNE]

    def _get_info_from_address(self, address):
        match = re.search(
            r'(?P<comune2>[^,]+, )?(?P<comune1>[^,]+), (?P<provincia>[^,]+), (?P<regione>[^,0-9]+)(?P<cap>, [0-9]{5})?, Italia',
            address)
        if not match:
            return None
        comune = match.group("comune1")
        comune = self._check_if_comune_is_comune(comune)
        if comune is None:
            comune = match.group("comune2")
            if comune is not None:
                comune = comune[:-2]
                comune = self._check_if_comune_is_comune(comune)
        if comune is None:
            comune = match.group("provincia")
            comune = comune.replace("Roma Capitale", "Roma")
            comune = self._check_if_comune_is_comune(comune)
        if comune is not None:
            comune = _clean_denom_text_value(comune)
        return comune

    def _check_matched_comune(self, row, value, match_dict):
        # Check if new value is/was a comune
        info_new_value = get_geo_info_from_comune(comune=value, flag_find_frazioni=False)

        if info_new_value is None:
            log.debug(f"Value {value} is/was not a comune.")
            return match_dict

        replace_pos = (self.df[self.MATCH_COLUMN] == row[self.MATCH_COLUMN])

        # If possible check if provincia or regione matched
        if cfg.TAG_CODICE_PROVINCIA in row:
            if row[cfg.TAG_CODICE_PROVINCIA] != info_new_value[cfg.TAG_CODICE_PROVINCIA]:
                log.debug(f"Value {value} is a comune, but the provincia not matched: "
                              f"{info_new_value[cfg.TAG_CODICE_PROVINCIA]} != {row[cfg.TAG_CODICE_PROVINCIA]}.")
                return match_dict
            else:
                provincia_column = self.detail_level[GeoLevel.PROVINCIA][0]
                replace_pos = replace_pos & (self.df[provincia_column] == row[provincia_column])

        if cfg.TAG_CODICE_REGIONE in row:
            if row[cfg.TAG_CODICE_REGIONE] != info_new_value[cfg.TAG_CODICE_REGIONE]:
                log.debug(f"Value {value} is a comune, but the regione not matched: "
                              f"{info_new_value[cfg.TAG_CODICE_REGIONE]} != {row[cfg.TAG_CODICE_REGIONE]}.")
                return match_dict
            elif cfg.TAG_CODICE_PROVINCIA not in row:
                regione_column = self.detail_level[GeoLevel.REGIONE][0]
                replace_pos = replace_pos & (self.df[regione_column] == row[regione_column])
        self.df.loc[replace_pos, self.MATCH_COLUMN] = value
        match_dict[row[self.MATCH_COLUMN]] = value
        return match_dict

    def run_find_frazioni(self):
        geo_level = list(self.detail_level.keys())[0]
        geo_code = list(self.detail_level.values())[0][1]

        # Checks
        if (geo_code != CodeLevel.DENOMINATION) or (geo_level != GeoLevel.COMUNE):
            raise Exception("Run find_frazioni only to find comune by denomination.")
        if self.not_match is None:
            raise Exception("Run simple match before.")

        n = self.get_n_not_matched()
        log.info(f"Needed to find {n} distinct values, we will use the open source api Nominatim that need 1 second "
                 f"between each call so it will need at least {n} seconds.")

        # Get the regione and/or provincia name for help and check for the results
        self._add_provincia_regione_denomination_to_not_matched()

        geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        match_dict = {}
        for index, row in self.not_match.iterrows():
            if cfg.TAG_PROVINCIA + self.SUFFIX_DEFAULT in row:
                detail = f", {row[cfg.TAG_PROVINCIA + self.SUFFIX_DEFAULT]}" or ""
            elif cfg.TAG_REGIONE + self.SUFFIX_DEFAULT in row:
                detail = f", {row[cfg.TAG_REGIONE + self.SUFFIX_DEFAULT]}" or ""
            else:
                detail = ""
            value = row[self.MATCH_COLUMN]
            location = geocode(f"{value}{detail}, italia")
            # Check if location is not None
            if (location is None) or (location.raw.get("addresstype") not in [
                "town", "village", "neighbourhood", "city", "district", "locality", "quarter", "subdistrict",
            ]):
                log.debug("Location not found.")
                continue
            location = location.address
            # Check if value in location
            if value not in location.lower().split(", "):
                log.debug(f"Location found but value {value} wasn't inside the address {location}.")
                continue

            comune = self._get_info_from_address(location)

            match_dict = self._check_matched_comune(row, comune, match_dict)

        if len(match_dict) > 0:
            log.info(f"Match {len(match_dict)} name that corresponds to a possible frazione of a comune:\n{match_dict}")
            _ = self._check_non_match(self.istat_registry[self.MATCH_COLUMN].unique())

    @staticmethod
    def _run_query(query, n_url_read):
        try:
            # First Try
            return list(
                search(query, tld='com', num=n_url_read, lang="it", country="Italy", stop=n_url_read, pause=2.5,
                       verify_ssl=False))
        except Exception as e:
            log.error('Failed to search on google tentative 1: ' + str(e))
            try:
                results = google_query(query, cfg.google_search_api_key, cfg.google_search_cse_id, num=n_url_read )
                return [result['link'] for result in results]
            except Exception as e:
                log.error('Failed to search on google tentative 2: ' + str(e))
                return []

    def _find_info_on_page(self, url, denomination):
        res = requests.get(url, verify=False)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        blacklist = [
            '[document]',
            'noscript',
            'header',
            'html',
            'meta',
            'head',
            'input',
            'script',
        ]
        sentences = " ".join([tag.string for tag in soup.find_all(text=True) if tag.parent.name not in blacklist])
        sentences = _clean_htmltext(sentences)
        sentences = re.split(r'[\r\n\.]', sentences)
        matches = []
        for text in sentences:
            results = re.findall(cfg.regex_find_frazioni.format(denomination), text)
            for result in results:
                _match = result[8].split("provincia")[0]
                _match = [_clean_denom_text_value(comune) for comune in self.istat_registry[cfg.TAG_COMUNE].unique() if re.match(f"\\b{comune.lower()}\\b", _match)]
                matches.extend(_match)
        matches = list(set(matches))
        return matches

    @validate
    def run_find_frazioni_on_web(self, n_url_read: int = 1):
        geo_level = list(self.detail_level.keys())[0]
        geo_code = list(self.detail_level.values())[0][1]

        # Checks
        if (geo_code != CodeLevel.DENOMINATION) or (geo_level != GeoLevel.COMUNE):
            raise Exception("Run run_find_frazioni_on_google only to find comune by denomination.")
        if self.not_match is None:
            raise Exception("Run simple match before.")

        n = self.get_n_not_matched()
        log.info(f"Needed to find {n} distinct values, we will search those on web that need 1 second "
                 f"between each call so it will need at least {n} seconds.")

        # Get the regione and/or provincia name for help and check for the results
        self._add_provincia_regione_denomination_to_not_matched()

        match_dict = {}
        for index, row in self.not_match.iterrows():
            if cfg.TAG_PROVINCIA + self.SUFFIX_DEFAULT in row:
                detail = f" {row[cfg.TAG_PROVINCIA + self.SUFFIX_DEFAULT]}" or ""
            elif cfg.TAG_REGIONE + self.SUFFIX_DEFAULT in row:
                detail = f" {row[cfg.TAG_REGIONE + self.SUFFIX_DEFAULT]}" or ""
            else:
                detail = ""
            value = row[self.MATCH_COLUMN]
            query = f""" "{value}" è una frazione del comune di {detail}"""

            # Search on web
            urls = self._run_query(query, n_url_read)

            for url in urls:
                match_comuni = self._find_info_on_page(url, value)
                for comune in match_comuni:
                    match_dict = self._check_matched_comune(row, comune, match_dict)
        if len(match_dict) > 0:
            log.info(f"Match {len(match_dict)} name that corresponds to a possible frazione of a comune from Web:\n{match_dict}")
            _ = self._check_non_match(self.istat_registry[self.MATCH_COLUMN].unique())

    @validate
    def get_result(
        self,
        add_missing: bool = False,
        drop_not_match: bool = False,
        suffix_result_columns: str = "",
        handle_duplicate_column: str = "error"
    ) -> pd.DataFrame:
        if self.not_match is None:
            raise Exception("Run simple match before get the result.")

        # Log results
        n_not_match = self.not_match[self.MATCH_COLUMN].nunique()
        if n_not_match > 0:
            log.warning(f"Unable to find {n_not_match} unique values: {self.get_not_matched_list()}")
        else:
            log.info(f"Found every values.")

        # Check column names in original dataset for any duplicates
        rename_columns = {
            col: col + suffix_result_columns
            for col in self.istat_registry.columns
            if col != self.MATCH_COLUMN
        }
        self.istat_registry.rename(columns=rename_columns, inplace=True)
        output_columns = list(set(self.OUTPUT_COLUMNS).intersection(self.istat_registry.columns))
        self.istat_registry = self.istat_registry[[self.MATCH_COLUMN] + output_columns]

        column_duplicates = list(set(output_columns).intersection(self.original_df.columns))

        if (len(column_duplicates) > 0) & (handle_duplicate_column == "error"):
            raise Exception(f"Found column in original dataset with the same name of one of the output columns:\n"
                            f"{column_duplicates}, change the 'handle_duplicate_column' params in order to handle "
                            f"those columns.\n'overwrite'= the original columns will be overwrite.\n"
                            f"suffix (str): this string will be used as suffix for the new columns.")
        elif (len(column_duplicates) > 0) & (handle_duplicate_column == "overwrite"):
            log.warning(f"Columns: {column_duplicates} will be overwrite from original dataset.")
            self.istat_registry.rename(
                columns={col: col + "geo_ita_rename_handler" for col in column_duplicates},
                inplace=True
            )
        elif (len(column_duplicates) > 0) & (isinstance(handle_duplicate_column, str)) & (handle_duplicate_column != ""):
            self.istat_registry.rename(
                columns={col: col + handle_duplicate_column for col in column_duplicates},
                inplace=True
            )

        join_columns = [name for name, code in self.detail_level.values()]

        if add_missing:
            if drop_not_match:
                how = "left"
            else:
                how = "outer"
            result = self.istat_registry.merge(self.df, on=self.MATCH_COLUMN, how=how)
            result = result.merge(self.original_df, on=join_columns, how=how)
        else:
            if drop_not_match:
                how = "inner"
            else:
                how = "left"
            result = self.df.merge(self.istat_registry, on=self.MATCH_COLUMN, how=how)
            result = result.merge(self.original_df, on=join_columns, how=how)

        if (len(column_duplicates) > 0) & (handle_duplicate_column == "overwrite"):
            result.drop(columns=[col + "geo_ita_rename_handler" for col in column_duplicates], inplace=True)
        result.drop(columns=[self.MATCH_COLUMN], inplace=True)
        return result

    @validate
    def run_similarity_match(self, unique_flag: bool = False, threshold=cfg.min_acceptable_similarity):
        if unique_flag:
            input_den = self.df[self.MATCH_COLUMN].values()
            registry_not_matched = [a for a in self.istat_registry[self.MATCH_COLUMN] if a not in input_den]
            match_dict = self._find_match(registry_not_matched, self.get_not_matched_list(), unique=True, threshold=threshold)
            self.similarity_result = {v[0]: (k, v[1]) for k, v in match_dict.items()}
        else:
            self.similarity_result = self._find_match(self.get_not_matched_list(), self.istat_registry[self.MATCH_COLUMN], threshold=threshold)
        n = len(self.similarity_result)
        if n > 1:
            log.info(f"Match {n} name by similarity:\n{self.similarity_result}")
        else:
            log.info("No match by similarity")

    def get_similarity_result(self):
        return self.similarity_result

    def accept_similarity_result(self):
        if self.similarity_result is not None:
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(
                {k: v[0] for k, v in self.similarity_result.items()})
            _ = self._check_non_match(self.istat_registry[self.MATCH_COLUMN])
        else:
            raise Exception("Run 'run_similarity_match' before accept_similarity_result.")

    @validate
    def use_manual_match(self, manual_dict: Dict[str, str]):
        geo_level = list(self.detail_level.keys())[0]
        geo_code = list(self.detail_level.values())[0][1]

        # Checks
        if (geo_code != CodeLevel.DENOMINATION) or (geo_level != GeoLevel.COMUNE):
            raise Exception("Run use_manual_match only to find comune by denomination.")

        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(manual_dict)
        _ = self._check_non_match(self.istat_registry[self.MATCH_COLUMN])

    @staticmethod
    def _find_match(not_match1, not_match2, threshold, unique=False) -> Dict[str, str]:
        """
        Parameters
        ----------
        not_match1: Lista di nomi da abbinare ad un valore della lista not_match2
        not_match2: Lista dei nomi a cui abbinare un valore della lista not_match1

        Returns
        Restituisce un dizionario contenente per ogni parola di not_match1 la parola più simile di not_match2 con il
        relativo punteggio
        """
        match_dict = {}
        for a in not_match1:
            best_match = difflib.get_close_matches(a, not_match2, 1)
            if len(best_match) > 0:
                best_match = best_match[0]
                score = difflib.SequenceMatcher(None, a, best_match).ratio()
                if score > threshold:
                    match_dict[a] = (best_match, score)
                    if unique:
                        not_match2.remove(best_match)
        return match_dict


def _get_tag_anag(code, level):
    if level == GeoLevel.COMUNE:
        if code == CodeLevel.CODE:
            result = cfg.TAG_CODICE_COMUNE
        else:
            result = cfg.TAG_COMUNE
    elif level == GeoLevel.PROVINCIA:
        if code == CodeLevel.CODE:
            result = cfg.TAG_CODICE_PROVINCIA
        elif code == CodeLevel.SIGLA:
            result = cfg.TAG_SIGLA
        else:
            result = cfg.TAG_PROVINCIA
    elif level == GeoLevel.REGIONE:
        if code == CodeLevel.CODE:
            result = cfg.TAG_CODICE_REGIONE
        else:
            result = cfg.TAG_REGIONE
    else:
        raise Exception("Level UNKNOWN")
    return result


def _clean_denom_text_value(value):
    value = value.lower()  # All strig in lowercase
    value = re.sub(r'[^\w\s]', ' ', value)  # Remove non alphabetic characters
    value = value.strip()
    value = re.sub(r'\s+', ' ', value)
    value = cfg.comuni_exceptions.get(value, value)
    value = unidecode.unidecode(value)
    value = cfg.comuni_exceptions.get(value, value)
    for k, v in cfg.clear_denomination.items():
        value = value.replace(k, v)
    return value


def __find_coord_columns(df):
    column_list = df.columns
    flag_coord_found = False
    lat_tag = None
    long_tag = None
    tags_list = [("lat", "lon"),
                 ("lat", "lng"),
                 ("latitudine", "longitudine"),
                 ("latitude", "longitude")]
    for _tags in tags_list:
        for col in column_list:
            if col.lower() == _tags[0]:
                lat_tag = col
            elif col.lower() == _tags[1]:
                long_tag = col
        if (lat_tag is not None) & (long_tag is not None):
            flag_coord_found = True
            break
        else:
            lat_tag = None
            long_tag = None
    log.info("Found columns about coordinates: ({}, {})".format(lat_tag, long_tag))
    return flag_coord_found, lat_tag, long_tag


def __create_geo_dataframe(df0, lat_tag=None, long_tag=None):
    if isinstance(df0, gpd.GeoDataFrame):
        df = df0.copy()
        if df.crs['init'] is None:
            coord_system = __find_coordinates_system(df, geometry="geometry")
            df.crs = {'init': coord_system}
    elif isinstance(df0, pd.DataFrame):
        if lat_tag is None:
            flag_coord_found, lat_tag, long_tag = __find_coord_columns(df0)
        else:
            flag_coord_found = True
        if flag_coord_found:
            df = df0[df0[long_tag].notnull()]
            df = gpd.GeoDataFrame(
                df.drop([long_tag, lat_tag], axis=1), geometry=gpd.points_from_xy(df[long_tag], df[lat_tag]))
            # df.loc[(df[long_tag].isna()) | (df[lat_tag].isna()), "geometry"] = None
            coord_system = __find_coordinates_system(df, lat_tag, long_tag)
            df.crs = {'init': coord_system}
        elif "geometry" in df0.columns:
            df = gpd.GeoDataFrame(df0)
            coord_system = __find_coordinates_system(df0, geometry="geometry")
            df.crs = {'init': coord_system}
            log.info("Found geometry columns")
        else:
            raise Exception("The DataFrame must have a geometry attribute or lat-long.")

    else:
        raise Exception("You need to pass a Pandas DataFrame of GeoDataFrame.")
    return df


def __find_coordinates_system(df, lat=None, lon=None, geometry=None):
    n_test = min(100, df.shape[0])
    if n_test == 0:
        return "epsg:4326"
    test = df.sample(n=n_test)
    if isinstance(df, gpd.GeoDataFrame):
        pass
    elif geometry is not None:
        test = gpd.GeoDataFrame(test, geometry=geometry)
    elif lat is not None and lon is not None:
        test = gpd.GeoDataFrame(
            test, geometry=gpd.points_from_xy(test[lon], test[lat]))
    else:
        raise Exception("To find the coordinate System usa lat-lon or geometry")

    italy = _get_shape_italia()
    italy.crs = {'init': "epsg:32632"}
    italy = italy.to_crs({'init': "epsg:4326"})
    test_join = gpd.tools.sjoin(test, italy, op='within')

    if test_join.shape[0] / n_test >= 0.8:
        log.info("Found coord system: epsg:4326")
        return "epsg:4326"

    italy = italy.to_crs({'init': "epsg:32632"})
    test_join = gpd.tools.sjoin(test, italy, op='within')

    if test_join.shape[0] / n_test >= 0.8:
        log.info("Found coord system: epsg:32632")
        return "epsg:32632"

    italy = italy.to_crs({'init': "epsg:3857"})
    test_join = gpd.tools.sjoin(test, italy, op='within')

    if test_join.shape[0] / n_test >= 0.8:
        log.info("Found coord system: epsg:3857")
        return "epsg:3857"

    log.warning("Unable to find coord system so the default is used epsg:4326")

    return "epsg:4326"


# @validate
def get_geo_info_from_comune(comune: str, provincia: str = None, regione: str = None, flag_find_frazioni: bool = True) -> Dict[str, str]:
    df = pd.DataFrame(data=[[comune, provincia, regione]], columns=["comune", "provincia", "regione"])
    addInfo = AddGeographicalInfo(df)
    addInfo.set_comuni_tag("comune")
    if provincia:
        addInfo.set_province_tag("provincia")
    elif regione:
        addInfo.set_regioni_tag("regione")
    addInfo.run_simple_match()
    if flag_find_frazioni:
        addInfo.run_find_frazioni()
    df = addInfo.get_result()
    if df[cfg.TAG_COMUNE].values[0] != df[cfg.TAG_COMUNE].values[0]:
        log.warning(f"Unable to find the city {comune}")
        return None
    tag_list = [cfg.TAG_COMUNE, cfg.TAG_CODICE_COMUNE,
                cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_CODICE_PROVINCIA,
                cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE, cfg.TAG_AREA_GEOGRAFICA,
                cfg.TAG_SUPERFICIE, cfg.TAG_POPOLAZIONE]
    result_dict = {a: df[a].values[0] for a in tag_list}
    return result_dict


@validate
def get_geo_info_from_regione(regione: str) -> Dict[str, str]:
    df = pd.DataFrame(data=[[regione]], columns=["regione"])
    addInfo = AddGeographicalInfo(df)
    addInfo.set_regioni_tag("regione")
    addInfo.run_simple_match()
    addInfo.run_find_frazioni()
    df = addInfo.get_result()
    if pd.isna(df[cfg.TAG_REGIONE].values[0]):
        raise Exception(f"Unable to find the region {regione}")
    result_dict = {
        "regione": df[cfg.TAG_REGIONE].values[0],
        "area_geografica": df[cfg.TAG_AREA_GEOGRAFICA].values[0],
        "popolazione": df[cfg.TAG_POPOLAZIONE].values[0],
        "superficie": df[cfg.TAG_SUPERFICIE].values[0]
    }
    return result_dict


@validate
def get_geo_info_from_provincia(provincia: str, regione: str = None) -> Dict[str, str]:
    df = pd.DataFrame(data=[[provincia, regione]], columns=["provincia", "regione"])
    addInfo = AddGeographicalInfo(df)
    addInfo.set_province_tag("provincia")
    if regione:
        addInfo.set_regioni_tag("regione")
    addInfo.run_simple_match()
    addInfo.run_find_frazioni()
    df = addInfo.get_result()
    if df[cfg.TAG_PROVINCIA].values[0] is None:
        raise Exception(f"Unable to find the city {provincia}")
    result_dict = {
        "provincia": df[cfg.TAG_PROVINCIA].values[0],
        "sigla": df[cfg.TAG_SIGLA].values[0],
        "regione": df[cfg.TAG_REGIONE].values[0],
        "area_geografica": df[cfg.TAG_AREA_GEOGRAFICA].values[0],
        "popolazione": df[cfg.TAG_POPOLAZIONE].values[0],
        "superficie": df[cfg.TAG_SUPERFICIE].values[0]
    }
    return result_dict


@validate
def get_city_from_coordinates(df0: pd.DataFrame,
                              latitude_columns: str = None, longitude_columns: str = None) -> pd.DataFrame:
    if latitude_columns is not None:
        _test_column_in_dataframe(df0, latitude_columns)
    if longitude_columns is not None:
        _test_column_in_dataframe(df0, longitude_columns)
    df0["key_mapping"] = range(df0.shape[0])
    df = df0.copy()
    df_comuni = get_df_comuni()
    df_comuni = gpd.GeoDataFrame(df_comuni)
    df_comuni.crs = {'init': 'epsg:32632'}
    df_comuni = df_comuni.to_crs({'init': 'epsg:4326'})

    df = __create_geo_dataframe(df, lat_tag=latitude_columns, long_tag=longitude_columns)
    df = df[df["geometry"].notnull()]
    df = df[["key_mapping", "geometry"]].drop_duplicates()
    df["geometry"] = df["geometry"].centroid
    df = df.to_crs({'init': 'epsg:4326'})

    n_tot = df.shape[0]
    map_city = gpd.sjoin(df, df_comuni, op='within', how="left")

    missing = list(map_city[map_city[cfg.TAG_COMUNE].isna()]["geometry"].unique())
    if len(missing) == 0:
        log.info("Found the correct city for each point")
    else:
        log.warning("Unable to find the city for {} points: {}".format(len(missing), [(x.x, x.y) for x in missing]))
    map_city = map_city[["key_mapping", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]]
    index_name = df0.index.name
    if index_name is None:
        index_name = "index"
    result = df0.reset_index().merge(map_city, on=["key_mapping"], how="left").drop(["key_mapping"], axis=1).set_index(
        index_name)
    df0.drop(["key_mapping"], axis=1, inplace=True)
    return result


def __test_city_in_address(df, city_tag, address_tag):
    return df.apply(lambda x: x[city_tag].lower() in x[address_tag]
    if (x[address_tag] and x[city_tag])
    else False,
                    axis=1)


def _try_replace_abbreviation_on_google(df, n_url_read, geocode):
    log.info("Try to find abbreviated name")
    not_found = list(df.loc[df["location"].isna(), "address_search"].unique())
    for address in not_found:
        address_without_city = address.split(",")[0]
        m = re.search(r"^([^.]+) (([a-z]+\. ?)+) ?([^.]+)$", address_without_city)
        if m:
            prefix = m.group(1)
            if prefix in cfg.list_road_prefix:
                prefix = "(" + "|".join(cfg.list_road_prefix) + ")"
            suffix = m.group(4)
            abbreviations = m.group(2)
            abbreviations = "[a-z]+ ?".join(abbreviations.replace(" ", "").split("."))
            match = []
            urls = []
            try:
                urls = list(
                    search(address, tld='com', num=n_url_read, lang="it", country="Italy", stop=n_url_read, pause=2.5,
                           verify_ssl=False))
            except Exception as e:
                log.error('Failed to search on google tentative 1: ' + str(e))
                try:
                    my_results = google_query(address,
                                              cfg.google_search_api_key,
                                              cfg.google_search_cse_id,
                                              num=n_url_read
                                              )
                    for result in my_results:
                        urls.append(result['link'])
                except Exception as e:
                    log.error('Failed to search on google tentative 2: ' + str(e))
            if len(urls) > 0:
                for url in urls:
                    res = requests.get(url, verify=False)
                    html_page = res.content
                    soup = BeautifulSoup(html_page, 'html.parser')
                    text = soup.find_all(text=True)
                    output = ' '.join(text)
                    output = _clean_htmltext(output)
                    pattern = f"{prefix} ?{abbreviations} ?{suffix}"
                    r1 = re.search(pattern, output)
                    if r1 is not None:
                        match.append(r1.group())
            match = list(set(match))
            if len(match) == 1:
                pos_match = df["address_search"] == address
                df.loc[pos_match, "address_search"] = match[0]
                if "," in address:
                    s = match[0] + "," + ", ".join(address.split(",")[1:])
                else:
                    s = match[0]
                location = geocode(s)
                if location is not None:
                    df.loc[pos_match, "latitude"] = location.latitude
                    df.loc[pos_match, "longitude"] = location.longitude
                    df.loc[pos_match, "address_test"] = location.address.lower()
    return df


def _try_wrong_replace_of_apostrophe(df, address_tag, geocode):
    log.info("Try to find wrong replace of apostrophe")
    regex = r"\b(del|dell|d|nell|sull|sant|Sant)([A-Z][^\s]+)"
    pos_replace = (df["location"].isna() &
                   df[address_tag].str.contains(regex))
    not_found = list(df.loc[pos_replace, address_tag].unique())
    for address in not_found:
        new_name = re.sub(regex, r"\1'\2", address)
        pos_match = df[address_tag] == address
        df.loc[pos_match, "address_search"] = new_name
        location = geocode(new_name)
        if location is not None:
            df.loc[pos_match, "latitude"] = location.latitude
            df.loc[pos_match, "longitude"] = location.longitude
            df.loc[pos_match, "address_test"] = location.address.lower()
    return df


def _find_location_with_openstreetmap(df, geocode):
    log.info(f"Run search on OpenStreetMap. Needed at least {df.shape[0]} seconds")
    start = datetime.now()
    df["location"] = (df["address_search"]).apply(geocode)
    df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
    df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)
    df["address_test"] = df["location"].apply(lambda loc: loc.address if loc else None).str.lower()
    log.info("Finding locations from address ended in {} seconds".format(datetime.now() - start))
    return df


def _test_address_with_comune_provincia_regione(df, comuni_tag, province_tag, regioni_tag):
    n_tot = df.shape[0]
    n_found = df["latitude"].notnull().sum()
    n_not_found = n_tot - n_found
    df["test"] = False
    if comuni_tag:
        df["test"] = __test_city_in_address(df, comuni_tag, "address_test")
        df.loc[~df["test"], "latitude"] = None
        df.loc[~df["test"], "longitude"] = None
        test_fail = (~df["test"]).sum() - n_not_found
        log.info(f"Found {n_found} location over {n_tot} address. But {test_fail} are not in the correct city.")
    elif province_tag:
        df["test"] = df["test"] | __test_city_in_address(df, province_tag, "address_test")
        df.loc[~df["test"], "latitude"] = None
        df.loc[~df["test"], "longitude"] = None
        test_fail = (~df["test"]).sum() - n_not_found
        log.info(f"Found {n_found} location over {n_tot} address. But {test_fail} are not in the correct provincia.")
    elif regioni_tag:
        df["test"] = df["test"] | __test_city_in_address(df, regioni_tag, "address_test")
        df.loc[~df["test"], "latitude"] = None
        df.loc[~df["test"], "longitude"] = None
        test_fail = (~df["test"]).sum() - n_not_found
        log.info(f"Found {n_found} location over {n_tot} address. But {test_fail} are not in the correct regione.")
    else:
        log.info(f"Found {n_found} location over {n_tot} address.")
    return df


@validate
def get_coordinates_from_address(df0: pd.DataFrame, address_tag: str,
                                 comuni_tag: str = None, province_tag: str = None, regioni_tag: str = None,
                                 n_url_read: int = 1) -> pd.DataFrame:
    # TODO add successive tentative (maps api)
    _test_column_in_dataframe(df0, address_tag)
    if comuni_tag is not None:
        _test_column_in_dataframe(df0, comuni_tag)
    if province_tag is not None:
        _test_column_in_dataframe(df0, province_tag)
    if regioni_tag is not None:
        _test_column_in_dataframe(df0, regioni_tag)

    col_list = [address_tag, comuni_tag, province_tag, regioni_tag]
    col_list = [x for x in col_list if x is not None]
    df = df0[col_list].drop_duplicates()

    df["address_search"] = df[address_tag].str.lower()
    if comuni_tag:
        t = __test_city_in_address(df, comuni_tag, "address_search")
        t = t | df[comuni_tag].isna()
        df["address_search"] = np.where(t, df["address_search"],
                                        df["address_search"] + ", " + df[comuni_tag].str.lower())

    geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    df = _find_location_with_openstreetmap(df, geocode)

    not_found_pos = df["location"].isna()

    if not_found_pos.sum() > 0:
        df = _try_replace_abbreviation_on_google(df, n_url_read, geocode)
        df = _try_wrong_replace_of_apostrophe(df, address_tag, geocode)

    df = _test_address_with_comune_provincia_regione(df, comuni_tag, province_tag, regioni_tag)

    # drop columns
    df.drop(["address_search", "location", "address_test", "test"], axis=1, inplace=True)
    # Join df0
    df = df0.merge(df, how="left", on=col_list)
    return df


@validate
def get_address_from_coordinates(df0: pd.DataFrame,
                                 latitude_columns: str = None, longitude_columns: str = None) -> pd.DataFrame:
    if latitude_columns is None or longitude_columns is None:
        flag_coord_found, latitude_columns, longitude_columns = __find_coord_columns(df0)
        if not flag_coord_found:
            raise Exception(
                "Unable to find the latitude and longitude columns. Please specify them in latitude_columns and longitude_columns")
    try:
        df0[latitude_columns] = df0[latitude_columns].astype(float)
        df0[latitude_columns] = df0[latitude_columns].astype(float)
    except:
        raise Exception("Use columns with float type for coordinates.")
    if (latitude_columns is not None and latitude_columns not in df0.columns) or \
            (longitude_columns is not None and longitude_columns not in df0.columns):
        raise Exception(
            "Use latitude_columns and longitude_column to specify the columns where to find the coordinates.")
    df = df0[[latitude_columns, longitude_columns]].drop_duplicates()
    df['geom'] = df[latitude_columns].map(str) + ', ' + df[longitude_columns].map(str)
    n = df.shape[0]
    log.info(f"Needed at least {n} seconds")
    geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    start = datetime.now()
    df["location"] = (df["geom"]).apply(reverse)
    log.info("Finding address from coordinates ended in {} seconds".format(datetime.now() - start))

    address_col = "address"
    if address_col in df0.columns:
        address_col = "address_geo_ita"
    city_col = "city"
    if city_col in df0.columns:
        city_col = "city_geo_ita"
    # TODO Ripulire address estraendo solo informazioni utili e uniformi
    df[address_col] = df["location"].apply(lambda loc: loc.address if loc else None).str.lower()
    df[city_col] = df["location"].apply(
        lambda loc: loc.raw["address"]["city"] if (loc and "city" in loc.raw["address"]) else None)
    df = df.drop(["geom", "location"], axis=1)
    ## Join df0
    df = df0.merge(df, how="left", on=[latitude_columns, longitude_columns])
    return df


def _distance_to_range_ccord(d):
    a = (14, 6)
    max_value = 1
    min_value = 0
    x_dist = 1
    while True:
        b = (a[0] + x_dist, a[1])
        dist = distance(a, b).m
        if dist == d:
            break
        elif dist > d:
            d1 = round((x_dist + min_value) / 2, 7)
            max_value = x_dist
            if d1 == x_dist:
                break
            else:
                x_dist = d1
        else:
            d1 = round((x_dist + max_value) / 2, 7)
            min_value = x_dist
            if d1 == x_dist:
                break
            else:
                x_dist = d1

    max_value = 1
    min_value = 0
    y_dist = 1
    while True:
        b = (a[0], a[1] + y_dist)
        dist = distance(a, b).m
        if dist == d:
            break
        elif dist > d:
            d1 = round((y_dist + min_value) / 2, 7)
            max_value = y_dist
            if d1 == y_dist:
                break
            else:
                y_dist = d1
        else:
            d1 = round((y_dist + max_value) / 2, 7)
            min_value = y_dist
            if d1 == y_dist:
                break
            else:
                y_dist = d1
    return x_dist, y_dist


@validate
def aggregate_point_by_distance(df0: pd.DataFrame,
                                distance_in_meters: Union[int, float],
                                latitude_columns: str = None, longitude_columns: str = None,
                                agg_column_name: str = "aggregation_code") -> pd.DataFrame:
    if latitude_columns is not None:
        _test_column_in_dataframe(df0, latitude_columns)
    if longitude_columns is not None:
        _test_column_in_dataframe(df0, longitude_columns)
    df0["key_mapping"] = range(df0.shape[0])
    df = __create_geo_dataframe(df0, latitude_columns, longitude_columns)
    df = df.to_crs({'init': 'epsg:4326'})
    df = df[["key_mapping", "geometry"]]
    df["geometry"] = df["geometry"].centroid
    radius_df = df.copy()
    # TODO Da rivedere (approssimazione distanza)
    x_dist, y_dist = _distance_to_range_ccord(distance_in_meters)
    dist = (x_dist + y_dist) / 2
    radius_df["geometry"] = radius_df.apply(lambda x: x['geometry'].buffer(dist, cap_style=1), axis=1)
    radius_df = gpd.sjoin(df, radius_df, op='within', how="left")
    radius_df = radius_df[["key_mapping_left", "key_mapping_right"]]
    n_points = df.shape[0]
    n_cc, df[agg_column_name] = connected_components(
        csr_matrix((np.ones(radius_df.shape[0]),
                    (radius_df["key_mapping_left"].values, radius_df["key_mapping_right"].values)),
                   shape=(n_points, n_points)),
        directed=False)

    df = df.set_index("key_mapping")[agg_column_name]
    df0[agg_column_name] = df0["key_mapping"].map(df)
    df0.drop(["key_mapping"], axis=1, inplace=True)
    log.info("The {} points have been aggregated in {} group. The largest has {} points.".format(df0.shape[0], n_cc,
                                                                                                 df0[
                                                                                                     agg_column_name].value_counts().values[
                                                                                                     0]))
    return df0


class GeoDataQuality:
    @validate
    def __init__(self, df: pd.DataFrame):
        self.original_df = df
        self.keys = None
        self.comuni_tag = None
        self.comuni_code = None
        self.comuni_result_tag = cfg.TAG_COMUNE
        self.province_tag = None
        self.province_code = None
        self.province_result_tag = cfg.TAG_PROVINCIA
        self.regioni_tag = None
        self.regioni_code = None
        self.regioni_result_tag = cfg.TAG_REGIONE
        self.nazione_tag = None
        self.latitude_tag = None
        self.longitude_tag = None
        self.check_tag = "_check"
        self.propose_tag = "_suggestion"
        self.flag_in_italy = "is_in_italy"
        self.sensitive = None

    @validate
    def set_keys(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        if not self.original_df[col_name].is_unique:
            raise Exception(r"Insert a column with unique values.")
        self.keys = col_name

    @validate
    def set_nazione_tag(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        self.nazione_tag = col_name

    @validate
    def set_regioni_tag(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        self.regioni_tag = col_name
        self.regioni_code = infer_geographical_category(list(self.original_df[col_name].unique()))
        self.regioni_result_tag = _get_tag_anag(self.regioni_code, cfg.LEVEL_REGIONE)

    @validate
    def set_comuni_tag(self, col_name: str, use_for_check_nation: bool = False):
        self.use_for_check_nation = use_for_check_nation
        _test_column_in_dataframe(self.original_df, col_name)
        self.comuni_tag = col_name
        self.comuni_code = infer_geographical_category(list(self.original_df[col_name].unique()))
        self.comuni_result_tag = _get_tag_anag(self.comuni_code, cfg.LEVEL_COMUNE)

    @validate
    def set_province_tag(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        self.province_tag = col_name
        self.province_code = infer_geographical_category(list(self.original_df[col_name].unique()))
        self.province_result_tag = _get_tag_anag(self.province_code, cfg.LEVEL_PROVINCIA)

    @validate
    def set_latitude_longitude_tag(self, lat_col: str, long_col: str):
        _test_column_in_dataframe(self.original_df, lat_col)
        _test_column_in_dataframe(self.original_df, long_col)
        self.latitude_tag = lat_col
        self.longitude_tag = long_col

    def _check_nazione(self):
        self.original_df[self.nazione_tag] = self._clean_denomination(self.original_df[self.nazione_tag])
        itali_string_names = ["it", "italy", "italia"]
        self._check_missing_values(self.nazione_tag)
        self.original_df[self.flag_in_italy] = self.original_df[self.nazione_tag].str.lower().isin(itali_string_names)
        if self.original_df[self.flag_in_italy].sum() > 0:
            values = self.original_df.loc[self.original_df[self.flag_in_italy], self.nazione_tag].value_counts()
            if values.shape[0] >= 1:
                self.italy_name = values.index[0]
                wrong_positions = (self.original_df[self.nazione_tag] != self.italy_name) & self.original_df[
                    self.flag_in_italy]

                self.original_df[self.nazione_tag + self.check_tag] = self.original_df[
                                                                          self.nazione_tag + self.check_tag] | wrong_positions
                self.original_df.loc[wrong_positions, self.nazione_tag + self.propose_tag] = self.italy_name

    def _clean_denomination(self, series):
        if not self.sensitive:
            series = series.str.lower()  # All strig in lowercase
            series = series.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non alphabetic characters
            series = series.str.strip()
            series = series.str.replace(r'\s+', ' ', regex=True)
            series = series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode(
                'utf-8')  # Remove accent
        return series

    def _check_regione(self):
        self.original_df[self.regioni_tag] = self._clean_denomination(self.original_df[self.regioni_tag])
        self._check_missing_values(self.regioni_tag)
        addinfo = AddGeographicalInfo(self.original_df)
        addinfo.set_regioni_tag(self.regioni_tag)
        addinfo.run_simple_match()
        check_df = addinfo.get_result()
        check_df[cfg.TAG_REGIONE] = self._clean_denomination(check_df[cfg.TAG_REGIONE])
        tag = self.regioni_result_tag
        self.original_df[tag + "_regione"] = check_df[tag]
        not_found_position = check_df[tag].isna() & self.original_df[self.flag_in_italy]
        self.original_df[self.regioni_tag + self.check_tag] = self.original_df[
                                                                  self.regioni_tag + self.check_tag] | not_found_position
        wrong_position = check_df[tag].notnull() & (check_df[tag] != check_df[self.regioni_tag])
        self.original_df.loc[wrong_position, self.regioni_tag + self.check_tag] = True
        self.original_df.loc[wrong_position, self.regioni_tag + self.propose_tag] = check_df.loc[wrong_position, tag]

    def _check_regione_nazione(self):
        pos = (self.original_df[self.nazione_tag].isna() | (self.original_df[self.nazione_tag] != self.italy_name)) & (
            self.original_df[self.regioni_result_tag + "_regione"].notnull())
        self.original_df.loc[pos, self.nazione_tag + self.check_tag] = True
        self.original_df.loc[pos, self.nazione_tag + self.propose_tag] = self.italy_name
        self.original_df.loc[pos, self.flag_in_italy] = True

    def _check_provincia(self):
        self.original_df[self.province_tag] = self._clean_denomination(self.original_df[self.province_tag])
        self._check_missing_values(self.province_tag)
        addinfo = AddGeographicalInfo(self.original_df)
        addinfo.set_province_tag(self.province_tag)
        addinfo.run_simple_match()
        check_df = addinfo.get_result()
        check_df[cfg.TAG_REGIONE] = self._clean_denomination(check_df[cfg.TAG_REGIONE])
        check_df[cfg.TAG_PROVINCIA] = self._clean_denomination(check_df[cfg.TAG_PROVINCIA])
        check_df[cfg.TAG_SIGLA] = self._clean_denomination(check_df[cfg.TAG_SIGLA])
        tag = self.province_result_tag
        self.original_df[self.regioni_result_tag + "_provincia"] = check_df[self.regioni_result_tag]
        self.original_df[tag + "_provincia"] = check_df[tag]
        not_found_position = check_df[tag].isna() & self.original_df[self.flag_in_italy]
        self.original_df.loc[not_found_position, self.province_tag + self.check_tag] = True
        wrong_position = check_df[tag].notnull() & (check_df[tag] != check_df[self.province_tag]) & (
                check_df[tag] != check_df[self.province_tag])
        self.original_df.loc[wrong_position, self.province_tag + self.check_tag] = True
        self.original_df.loc[wrong_position, self.province_tag + self.propose_tag] = check_df.loc[
            wrong_position, tag]

    def _check_provincia_nazione(self):
        pos = (self.original_df[self.nazione_tag].isna() | (self.original_df[self.nazione_tag] != self.italy_name)) & (
            self.original_df[self.regioni_result_tag + "_provincia"].notnull())
        self.original_df.loc[pos, self.nazione_tag + self.check_tag] = True
        self.original_df.loc[pos, self.nazione_tag + self.propose_tag] = self.italy_name
        self.original_df.loc[pos, self.flag_in_italy] = True

    def _check_provincia_regione(self):
        pos = self.original_df[self.regioni_result_tag + "_provincia"].notnull() & self.original_df[
            self.regioni_result_tag + "_regione"].isna()
        self.original_df.loc[pos, self.regioni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.regioni_tag + self.propose_tag] = self.original_df.loc[
            pos, self.regioni_result_tag + "_provincia"]
        pos = self.original_df[self.regioni_result_tag + "_provincia"].notnull() & self.original_df[
            self.regioni_result_tag + "_regione"].notnull() & (
                      self.original_df[self.regioni_result_tag + "_provincia"] != self.original_df[
                  self.regioni_result_tag + "_regione"])
        self.original_df.loc[pos, self.regioni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.regioni_tag + self.propose_tag] = None

    def _check_comune(self):
        self._check_missing_values(self.comuni_tag)
        if self.use_for_check_nation:
            addinfo = AddGeographicalInfo(self.original_df)
        else:
            addinfo = AddGeographicalInfo(self.original_df[self.original_df[self.flag_in_italy]])
        addinfo.set_comuni_tag(self.comuni_tag)
        if self.province_tag:
            addinfo.set_province_tag(self.province_tag)
        addinfo.run_simple_match()
        # try:
        addinfo.run_find_frazioni()
        addinfo.run_find_frazioni_from_google()
        # except:
        #    pass
        check_df = addinfo.get_result()
        self.original_df[self.comuni_tag] = self._clean_denomination(self.original_df[self.comuni_tag])
        check_df[cfg.TAG_REGIONE] = self._clean_denomination(check_df[cfg.TAG_REGIONE])
        check_df[cfg.TAG_PROVINCIA] = self._clean_denomination(check_df[cfg.TAG_PROVINCIA])
        check_df[cfg.TAG_SIGLA] = self._clean_denomination(check_df[cfg.TAG_SIGLA])
        check_df[cfg.TAG_COMUNE] = self._clean_denomination(check_df[cfg.TAG_COMUNE])
        if self.use_for_check_nation:
            self.original_df[self.regioni_result_tag + "_comune"] = check_df[self.regioni_result_tag]
            self.original_df[self.province_result_tag + "_comune"] = check_df[self.province_result_tag]
            self.original_df[self.comuni_result_tag + "_comune"] = check_df[self.comuni_result_tag]
        else:
            self.original_df.loc[self.original_df[self.flag_in_italy], self.regioni_result_tag + "_comune"] = check_df[
                self.regioni_result_tag]
            self.original_df.loc[self.original_df[self.flag_in_italy], self.province_result_tag + "_comune"] = check_df[
                self.province_result_tag]
            self.original_df.loc[self.original_df[self.flag_in_italy], self.comuni_result_tag + "_comune"] = check_df[
                self.comuni_result_tag]
        not_found_position = check_df[self.comuni_result_tag].isna() & self.original_df[self.flag_in_italy]
        self.original_df.loc[not_found_position, self.comuni_tag + self.check_tag] = True
        wrong_position = (self.original_df[self.comuni_result_tag + "_comune"] != self.original_df[self.comuni_tag]) & \
                         self.original_df[self.comuni_result_tag + "_comune"]
        self.original_df.loc[wrong_position, self.comuni_tag + self.check_tag] = True
        self.original_df.loc[wrong_position, self.comuni_tag + self.propose_tag] = check_df.loc[
            wrong_position, self.comuni_result_tag]

    def _check_comune_nazione(self):
        pos = (self.original_df[self.nazione_tag].isna() | (self.original_df[self.nazione_tag] != self.italy_name)) & (
            self.original_df[self.regioni_result_tag + "_comune"].notnull())
        self.original_df.loc[pos, self.nazione_tag + self.check_tag] = True
        self.original_df.loc[pos, self.nazione_tag + self.propose_tag] = self.italy_name
        self.original_df.loc[pos, self.flag_in_italy] = True

    def _check_comune_regione(self):
        pos = self.original_df[self.regioni_result_tag + "_comune"].notnull() & self.original_df[
            self.regioni_result_tag + "_regione"].isna() & self.original_df[self.regioni_tag + self.propose_tag].isna()
        self.original_df.loc[pos, self.regioni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.regioni_tag + self.propose_tag] = self.original_df.loc[
            pos, self.regioni_result_tag + "_comune"]
        pos = self.original_df[self.regioni_result_tag + "_comune"].notnull() & (
                (self.original_df[self.regioni_result_tag + "_regione"].notnull() &
                 (self.original_df[self.regioni_result_tag + "_comune"] != self.original_df[
                     self.regioni_result_tag + "_regione"])) |
                (self.original_df[self.regioni_tag + self.propose_tag].notnull() &
                 (self.original_df[self.regioni_result_tag + "_comune"] != self.original_df[
                     self.regioni_tag + self.propose_tag])))
        self.original_df.loc[pos, self.regioni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.regioni_tag + self.propose_tag] = None

    def _check_comune_provincia(self):
        pos = self.original_df[self.province_result_tag + "_comune"].notnull() & self.original_df[
            self.province_result_tag + "_provincia"].isna()
        self.original_df.loc[pos, self.province_tag + self.check_tag] = True
        self.original_df.loc[pos, self.province_tag + self.propose_tag] = self.original_df.loc[
            pos, self.province_result_tag + "_comune"]
        pos = self.original_df[self.province_result_tag + "_comune"].notnull() & self.original_df[
            self.province_result_tag + "_provincia"].notnull() & (
                      self.original_df[self.province_result_tag + "_comune"] != self.original_df[
                  self.province_result_tag + "_provincia"])
        self.original_df.loc[pos, self.province_tag + self.check_tag] = True
        self.original_df.loc[pos, self.province_tag + self.propose_tag] = None

    def _check_coordinates(self):
        check_tag = "coordinates" + self.check_tag
        self.original_df[check_tag] = (self.original_df[self.latitude_tag].isna() |
                                       self.original_df[self.longitude_tag].isna()) & self.original_df[
                                          self.flag_in_italy]
        check_df = get_city_from_coordinates(self.original_df, self.latitude_tag, self.longitude_tag)
        check_df[cfg.TAG_REGIONE] = self._clean_denomination(check_df[cfg.TAG_REGIONE])
        check_df[cfg.TAG_PROVINCIA] = self._clean_denomination(check_df[cfg.TAG_PROVINCIA])
        check_df[cfg.TAG_SIGLA] = self._clean_denomination(check_df[cfg.TAG_SIGLA])
        check_df[cfg.TAG_COMUNE] = self._clean_denomination(check_df[cfg.TAG_COMUNE])
        not_found_position = check_df[cfg.TAG_COMUNE].isna() & self.original_df[self.flag_in_italy]
        self.original_df.loc[not_found_position, check_tag] = True
        self.original_df[self.regioni_result_tag + "_coordinates"] = check_df[self.regioni_result_tag]
        self.original_df[self.province_result_tag + "_coordinates"] = check_df[self.province_result_tag]
        self.original_df[self.comuni_result_tag + "_coordinates"] = check_df[self.comuni_result_tag]

    def _check_coordinates_nazione(self):
        pos = (self.original_df[self.nazione_tag].isna() | (self.original_df[self.nazione_tag] != self.italy_name)) & (
            self.original_df[cfg.TAG_REGIONE + "_coordinates"].notnull())
        self.original_df.loc[pos, self.nazione_tag + self.check_tag] = True
        self.original_df.loc[pos, self.nazione_tag + self.propose_tag] = self.italy_name
        self.original_df.loc[pos, self.flag_in_italy] = True

    def _check_coordinates_regione(self):
        pos = self.original_df[self.regioni_result_tag + "_coordinates"].notnull() & self.original_df[
            self.regioni_result_tag + "_regione"].isna() & self.original_df[self.regioni_tag + self.propose_tag].isna()
        self.original_df.loc[pos, self.regioni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.regioni_tag + self.propose_tag] = self.original_df.loc[
            pos, self.regioni_result_tag + "_coordinates"]
        pos = self.original_df[self.regioni_result_tag + "_coordinates"].notnull() & (
                (self.original_df[self.regioni_result_tag + "_regione"].notnull() &
                 (self.original_df[self.regioni_result_tag + "_coordinates"] != self.original_df[
                     self.regioni_result_tag + "_regione"])) |
                (self.original_df[self.regioni_tag + self.propose_tag].notnull() &
                 (self.original_df[self.regioni_result_tag + "_coordinates"] != self.original_df[
                     self.regioni_tag + self.propose_tag])))
        self.original_df.loc[pos, self.regioni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.regioni_tag + self.propose_tag] = None

    def _check_coordinates_provincia(self):
        pos = self.original_df[self.province_result_tag + "_coordinates"].notnull() & self.original_df[
            self.province_result_tag + "_provincia"].isna() & self.original_df[
                  self.province_tag + self.propose_tag].isna()
        self.original_df.loc[pos, self.province_tag + self.check_tag] = True
        self.original_df.loc[pos, self.province_tag + self.propose_tag] = self.original_df.loc[
            pos, self.province_result_tag + "_coordinates"]
        pos = self.original_df[self.province_result_tag + "_coordinates"].notnull() & (
                (self.original_df[self.province_result_tag + "_provincia"].notnull() &
                 (self.original_df[self.province_result_tag + "_coordinates"] != self.original_df[
                     self.province_result_tag + "_provincia"])) |
                (self.original_df[self.province_tag + self.propose_tag].notnull() &
                 (self.original_df[self.province_result_tag + "_coordinates"] != self.original_df[
                     self.province_tag + self.propose_tag])))
        self.original_df.loc[pos, self.province_tag + self.check_tag] = True
        self.original_df.loc[pos, self.province_tag + self.propose_tag] = None

    def _check_coordinates_comune(self):
        pos = self.original_df[self.comuni_result_tag + "_coordinates"].notnull() & self.original_df[
            self.comuni_result_tag + "_comune"].isna()
        self.original_df.loc[pos, self.comuni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.comuni_tag + self.propose_tag] = self.original_df.loc[
            pos, cfg.TAG_COMUNE + "_coordinates"]
        pos = self.original_df[self.comuni_result_tag + "_coordinates"].notnull() & self.original_df[
            self.comuni_result_tag + "_comune"].notnull() & (
                      self.original_df[self.comuni_result_tag + "_coordinates"] != self.original_df[
                  self.comuni_result_tag + "_comune"])
        self.original_df.loc[pos, self.comuni_tag + self.check_tag] = True
        self.original_df.loc[pos, self.comuni_tag + self.propose_tag] = None

    @validate
    def start_check(self, show_only_warning: bool = True, sensitive: bool = False) -> pd.DataFrame:
        self.sensitive = sensitive
        col_list = [self.keys, self.nazione_tag, self.regioni_tag, self.province_tag, self.comuni_tag,
                    self.latitude_tag, self.longitude_tag]
        col_list = [a for a in col_list if a is not None]
        self.original_df = self.original_df[col_list].drop_duplicates()

        self.original_df[self.flag_in_italy] = True
        if self.nazione_tag is not None:
            self._check_nazione()

        if self.regioni_tag is not None:
            self._check_regione()
            if self.nazione_tag is not None:
                self._check_regione_nazione()

        if self.province_tag is not None:
            self._check_provincia()
            if self.nazione_tag is not None:
                self._check_provincia_nazione()
            if self.regioni_tag is not None:
                self._check_provincia_regione()

        if self.comuni_tag is not None:
            self._check_comune()
            if self.nazione_tag is not None:
                self._check_comune_nazione()
            if self.regioni_tag is not None:
                self._check_comune_regione()
            if self.province_tag is not None:
                self._check_comune_provincia()

        if self.latitude_tag is not None:
            self._check_coordinates()
            if self.nazione_tag is not None:
                self._check_coordinates_nazione()
            if self.regioni_tag is not None:
                self._check_coordinates_regione()
            if self.province_tag is not None:
                self._check_coordinates_provincia()
            if self.comuni_tag is not None:
                self._check_coordinates_comune()

        check_list = [self.nazione_tag, self.regioni_tag, self.province_tag, self.comuni_tag]
        if self.latitude_tag is not None:
            check_list.append("coordinates")
        check_list = [a + self.check_tag for a in check_list if a is not None]

        self.original_df["check"] = self.original_df[check_list].any(axis='columns')
        self.original_df["solved"] = np.where(self.original_df["check"], True, None)
        for c in check_list:
            pos = self.original_df[c]
            if c != ("coordinates" + self.check_tag):
                not_solved = self.original_df[c.replace(self.check_tag, self.propose_tag)].isna()
                self.original_df.loc[pos & not_solved, "solved"] = False
            else:
                self.original_df.loc[pos, "solved"] = False
        n_tot = self.original_df.shape[0]

        n_check = self.original_df["check"].sum()
        n_solved = self.original_df["solved"].sum()
        log.info("Found {} problems over {} ({}%), of which {} solved ({}%)".format(n_check, n_tot,
                                                                                    round(n_check / n_tot * 100, 1),
                                                                                    n_solved,
                                                                                    round(n_solved / n_check * 100, 1)))
        if self.nazione_tag:
            log.info("Field {}: {} problem, {} solved".format(
                self.nazione_tag, self.original_df[self.nazione_tag + self.check_tag].sum(),
                self.original_df[self.nazione_tag + self.propose_tag].notnull().sum()
            ))
        if self.regioni_tag:
            log.info("Field {}: {} problem, {} solved".format(
                self.regioni_tag, self.original_df[self.regioni_tag + self.check_tag].sum(),
                self.original_df[self.regioni_tag + self.propose_tag].notnull().sum()
            ))
        if self.province_tag:
            log.info("Field {}: {} problem, {} solved".format(
                self.province_tag, self.original_df[self.province_tag + self.check_tag].sum(),
                self.original_df[self.province_tag + self.propose_tag].notnull().sum()
            ))
        if self.comuni_tag:
            log.info("Field {}: {} problem, {} solved".format(
                self.comuni_tag, self.original_df[self.comuni_tag + self.check_tag].sum(),
                self.original_df[self.comuni_tag + self.propose_tag].notnull().sum()
            ))
        if self.latitude_tag:
            log.info("Coordinates: {} problem".format(
                self.original_df["coordinates" + self.check_tag].sum()
            ))

        if show_only_warning:
            result = self.original_df[self.original_df["check"]]
        else:
            result = self.original_df
        return result

    def _check_missing_values(self, col_name):
        self.original_df[col_name + self.check_tag] = self.original_df[col_name].isna() & self.original_df[
            self.flag_in_italy]

    @staticmethod
    def _create_header(width, background_color, text_color, title, subtitle):
        height = 100 if subtitle is not None else 50
        header = figure(x_range=(0, 1), y_range=(0, 1),
                        width=width, height=height,
                        tools="")
        header.background_fill_color = background_color
        header.xgrid.grid_line_color = None
        header.ygrid.grid_line_color = None
        header.axis.visible = False
        header.toolbar.logo = None
        header.outline_line_color = None
        header.toolbar_location = None
        header.min_border_left = 0
        header.min_border_right = 0
        header.min_border_top = 0
        header.min_border_bottom = 0

        header.add_layout(
            Label(x=0.005, y=.8, text=title,
                  text_font_style="bold",
                  text_font_size="20pt",
                  text_baseline="top",
                  text_color=text_color))

        if subtitle is not None:
            header.add_layout(
                Label(x=0.005, y=.4, text=subtitle,
                      text_font_size="12pt",
                      text_baseline="top",
                      text_color=text_color))
        return header

    def _create_map_plot(self, width, source_info):

        margins = [[723576.6901562785, 2070542.52875489], [4355801.264971882, 5999391.278141545]]

        map_plot = figure(x_range=(margins[0][0], margins[0][1]),
                          y_range=(margins[1][0], margins[1][1]),
                          x_axis_type="mercator", y_axis_type="mercator", width=width,
                          tools='pan,tap,wheel_zoom')
        map_plot.add_tile(xyz.CartoDB.Positron)
        map_plot.xgrid.grid_line_color = None
        map_plot.ygrid.grid_line_color = None
        map_plot.yaxis.visible = False
        map_plot.grid.visible = False
        map_plot.toolbar.logo = None
        map_plot.outline_line_color = None
        map_plot.xaxis.major_tick_line_color = None
        map_plot.xaxis.minor_tick_line_color = None
        map_plot.xaxis.major_label_text_font_size = '0pt'

        plot1 = map_plot.circle(x="longitudine_marcator", y="latitudine_marcator",
                                size=7,
                                fill_alpha="selected_alpha",
                                line_color="gray", line_width=0.5, source=source_info)
        # plot1 = map_plot.add_glyph(source_info, points)

        tooltips = [(self.keys, "@" + self.keys)]
        if self.nazione_tag is not None:
            tooltips.append(("Nazione", "@" + self.nazione_tag))
        if self.regioni_tag is not None:
            tooltips.append(("Regione", "@" + self.regioni_tag))
        if self.province_tag is not None:
            tooltips.append(("Provincia", "@" + self.province_tag))
        if self.comuni_tag is not None:
            tooltips.append(("Comune", "@" + self.comuni_tag))

        tooltips.append(
            ("Coordinates", "(@" + self.latitude_tag + "{0,0.0000000}-@" + self.longitude_tag + "{0,0.0000000})"))

        map_plot.add_tools(HoverTool(renderers=[plot1], tooltips=tooltips))

        map_plot.toolbar.active_scroll = map_plot.select_one(WheelZoomTool)

        return map_plot

    def _create_text_key_copy(self):
        text_input = TextInput(value="", title=self.keys + ": ", width=200)
        return text_input

    @validate
    def plot_result(self, background_color: str = "white", text_color: str = "black",
                    title: str = "Geographical DataQuality",
                    subtitle: str = None,
                    show_only_warning: bool = True,
                    save_in_path: Union[str, Path] = None):
        n_tot = self.original_df.shape[0]
        width = 1000
        width_check = 250
        row_height = 30
        perc_height = 50
        ok_tag = "OK"
        warning_tag = "Warning"
        solved_tag = "Warning solved"
        if show_only_warning:
            source = self.original_df[self.original_df["check"]]
        else:
            source = self.original_df
        n_plot = source.shape[0]
        height = (n_plot + 1) * row_height
        source["x"] = 0.5
        source["y"] = range(n_plot)[::-1]
        source["y"] += 0.5
        source["check_color"] = ok_tag
        source["selected_color"] = "transparent"
        source["selected_alpha"] = 1
        if self.latitude_tag:
            source['longitudine_marcator'], source['latitudine_marcator'] = transform(inProj, outProj,
                                                                                      source[
                                                                                          self.longitude_tag].tolist(),
                                                                                      source[
                                                                                          self.latitude_tag].tolist())
        pos = source["check"]
        source.loc[pos, "check_color"] = warning_tag
        pos = source["solved"]
        source.loc[pos, "check_color"] = solved_tag

        if self.keys is None:
            self.keys = "index"
            source = source.reset_index().rename(columns={source.index.name: self.keys})

        source[self.keys] = source[self.keys].astype(str)

        # source = source.where(pd.notnull(source), None)
        propose_col = [a for a in source.columns if self.propose_tag in a]
        source[propose_col] = source[propose_col].fillna("")

        col_list = [self.nazione_tag, self.regioni_tag, self.province_tag, self.comuni_tag]
        col_list = [a for a in col_list if a is not None]

        template = """
               <div style="background:<%= 
               (function colorfromint(){{
                        return(selected_color)
                            }}()) %>; 
                   color: black">
               <%= value %> 
               </div>
               """
        formatter = HTMLTemplateFormatter(template=template)
        columns = [TableColumn(field=self.keys, title=self.keys, formatter=formatter)]

        tag_mapping = {self.nazione_tag: (None, None),
                       self.regioni_tag: (self.regioni_result_tag, "Regione"),
                       self.province_tag: (self.province_result_tag, "Provincia"),
                       self.comuni_tag: (self.comuni_result_tag, "Comune")}

        perc_data = []
        legend_data = []
        html_tag = "_html"
        i = 1
        for c in col_list:
            n_check = (source[c + self.check_tag] & (source[c + self.propose_tag] == "")).sum()
            perc_data.append([i + 0.8, 0.75, "{} ({}%)".format(n_check, int(round(n_check / n_tot * 100, 0)))])
            n_propose = (source[c + self.propose_tag] != "").sum()
            perc_data.append([i + 0.8, 0.25, "{} ({}%)".format(n_propose, int(round(n_propose / n_tot * 100, 0)))])
            legend_data.append([i + 0.9, 0.75, warning_tag, c + self.check_tag])
            legend_data.append([i + 0.9, 0.25, solved_tag, c + self.check_tag])
            pos = source[c + self.propose_tag] != ""
            # source[c].fillna("NaN", inplace=True)
            original = source[c].fillna("NaN").copy()
            source[c + html_tag] = np.where(pos,
                                            original + "||" + source[c + self.propose_tag],
                                            " ||" + original)
            source[c + html_tag] = source[c + html_tag] + "||" + original + "||" + source[c + self.propose_tag]
            tag, name = tag_mapping[c]
            html_tooltip = ""
            if tag is not None:
                if tag + "_regione" in source.columns:
                    html_tooltip += "\n{} found from regione: <%= value.split('||')[4] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_regione"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
                if tag + "_provincia" in source.columns:
                    html_tooltip += "\n{} found from provincia: <%= value.split('||')[5] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_provincia"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
                if tag + "_comune" in source.columns:
                    html_tooltip += "\n{} found from comune: <%= value.split('||')[6] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_comune"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
                if tag + "_coordinates" in source.columns:
                    html_tooltip += "\n{} found from coordinates: <%= value.split('||')[7] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_coordinates"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
            template = """
                        <div style="background:<%= 
                            (function colorfromint(){{
                                if({check}){{
                                    if({propose} != ""){{
                                        return("orange")
                                        }} else {{
                                        return("red")
                                    }}
                                }}
                            }}()) %>; 
                            color: black">
                        <span href="#" data-toggle="tooltip" title="Original: <%= value.split('||')[2] %>\nSuggestion: <%= value.split('||')[3] %>{html_tooltip}">
                            <strike><%=  value.split("||")[0] %></strike> <%= value.split("||")[1] %>
                        </span>
                        </div>
                        """.format(check=c + self.check_tag,
                                   propose=c + self.propose_tag,
                                   html_tooltip=html_tooltip)
            formatter = HTMLTemplateFormatter(template=template)
            columns.append(TableColumn(field=c + html_tag, title=c, formatter=formatter))
            i += 1

        if self.latitude_tag:
            n_check = source["coordinates" + self.check_tag].sum()
            perc_data.append([i + 0.8, 0.75, "{} ({}%)".format(n_check, int(round(n_check / n_tot * 100, 0)))])
            legend_data.append([i + 0.9, 0.75, warning_tag, "coordinates" + self.check_tag])
            i += 1
            perc_data.append([i + 0.8, 0.75, "{} ({}%)".format(n_check, int(round(n_check / n_tot * 100, 0)))])
            legend_data.append([i + 0.9, 0.75, warning_tag, "coordinates" + self.check_tag])
            i += 1
            template = """
                        <div style="background:<%= 
                            (function colorfromint(){{
                                if({}){{
                                    return("red")}}
                                }}()) %>; 
                            color: black"> 
                        <%= value %>
                        </div>
                        """.format("coordinates" + self.check_tag)
            formatter = HTMLTemplateFormatter(template=template)
            columns.append(
                TableColumn(field=self.latitude_tag, title=self.latitude_tag, formatter=formatter))
            columns.append(
                TableColumn(field=self.longitude_tag, title=self.longitude_tag, formatter=formatter))

        header = self._create_header(width + width_check, background_color, text_color, title, subtitle)
        text_input = self._create_text_key_copy()
        column_drop = ["is_in_italy", self.regioni_tag + "_regione", self.regioni_tag + "_provincia",
                       self.regioni_tag + "_comune", self.regioni_tag + "_coordinates",
                       self.province_tag + "_provincia", self.province_tag + "_comune",
                       self.province_tag + "_coordinates",
                       self.comuni_tag + "_comune", self.comuni_tag + "_coordinates"]
        column_drop = [a for a in column_drop if a in source.columns]
        source = source.drop(column_drop, axis=1)
        originalsource = ColumnDataSource(source)
        source = ColumnDataSource(source)
        data_table = DataTable(source=source,
                               columns=columns,
                               fit_columns=True,
                               selectable=True,
                               sortable=False,
                               editable=True,
                               index_position=None,
                               row_height=row_height,
                               height=height, width=width)

        source.selected.js_on_change('indices',
                                     CustomJS(args=dict(source=source,
                                                        text=text_input), code="""
                            var indices = cb_obj.indices;
                            console.log(cb_obj)
                            if (indices.length > 0){{

                                var current_value = text.value; 
                                var pos = cb_obj.indices[0];
                                console.log("Selected", pos) 
                                var data = source.data;

                                var selected_value = data["{key}"][pos];
                                if (current_value != selected_value){{
                                    for (var i = 0; i < data["{key}"].length; i++) {{
                                        data["selected_alpha"][i] = 0;
                                    }}
                                }} else {{
                                    for (var i = 0; i < data["{key}"].length; i++) {{
                                        data["selected_alpha"][i] = 1;
                                    }}
                                }}
                                if (current_value == "") {{
                                    text.value = selected_value;
                                    data["selected_color"][pos] = "yellow";
                                    data["selected_alpha"][pos] = 1;
                                }} else {{
                                    for (var i = 0; i < data["{key}"].length; i++) {{
                                        data["selected_color"][i] = "transparent";
                                    }}
                                    if (current_value != selected_value) {{
                                        text.value = selected_value;
                                        data["selected_color"][pos] = "yellow";
                                        data["selected_alpha"][pos] = 1;
                                    }} else {{
                                        text.value = "";
                                    }}
                                }}
                                source.change.emit();
                            }}
                            console.log("Ended")
                            cb_obj.indices = [];
                        """.format(key=self.keys))
                                     )

        check_plot = figure(
            height=height,
            width=width_check,
            x_range=(0, 1),
            y_range=(0, n_plot),
            x_axis_location="above",
            tools='')
        check_plot.xgrid.grid_line_color = None
        check_plot.ygrid.grid_line_color = None
        check_plot.yaxis.visible = False
        check_plot.grid.visible = False
        check_plot.toolbar.logo = None
        check_plot.outline_line_color = None
        check_plot.xaxis.major_label_text_font_size = '10pt'
        check_plot.xaxis.ticker = [0.5]
        check_plot.xaxis.major_label_overrides = {0.5: "Check"}

        check_plot.circle(x="x", y="y", size=9, line_width=0.5,
                          fill_color={"field": "check_color",
                                      "transform": CategoricalColorMapper(factors=[ok_tag, solved_tag, warning_tag],
                                                                          palette=["green", "orange", "red"])},
                          source=source, legend_label="check_color")
        check_plot.add_layout(check_plot.legend[0], 'right')

        perc_plot = figure(
            height=perc_height,
            width=width,
            x_range=(0, i),
            y_range=(0, 1),
            tools='tap')
        perc_plot.title.text_font_size = '16pt'
        perc_plot.xgrid.grid_line_color = None
        perc_plot.ygrid.grid_line_color = None
        perc_plot.yaxis.visible = False
        perc_plot.xaxis.visible = False
        perc_plot.grid.visible = False
        perc_plot.toolbar.logo = None
        perc_plot.outline_line_color = None
        perc_plot.xaxis.major_tick_line_color = None
        perc_plot.xaxis.minor_tick_line_color = None
        perc_plot.xaxis.major_label_text_font_size = '0pt'
        perc_plot.toolbar_location = None
        legend_data = np.array(legend_data)
        legend_data = ColumnDataSource(dict(
            x=legend_data[:, 0].astype(float),
            y=legend_data[:, 1].astype(float),
            color=legend_data[:, 2],
            column=legend_data[:, 3],
            alpha=np.ones(legend_data.shape[0]) * 0.5))

        legend_data.selected.js_on_change('indices',
                                          CustomJS(args=dict(source=source,
                                                             original_source=originalsource,
                                                             legend_source=legend_data), code="""
                                    var indices = cb_obj.indices;
                                    if (indices.length > 0){
                                        var df_legend = legend_source.data;
                                        var pos = cb_obj.indices[0];
                                        console.log("Selected", pos) 
                                        var data = source.data;
                                        var column_selected = df_legend["column"][pos];
                                        var color_selected = df_legend["y"][pos];
                                        var previus_selected = (df_legend["alpha"][pos] == 1);
                                        console.log("Previus selected", previus_selected) 
                                        var df0 = original_source.data;
                                        var df = source.data;
                                        if (previus_selected){
                                            df_legend["alpha"][pos] = 0.5
                                            for (var key in df0) {
                                                df[key] = [];
                                                for (var i = 0; i < df0[key].length; ++i) {
                                                    df[key].push(df0[key][i]);
                                                }
                                            }
                                        } else {
                                            df_legend["alpha"][pos] = 1
                                            for (var key in df0) {
                                                var y_val = df0[key].length + 0.5
                                                df[key] = [];
                                                for (i = 0; i < df0[key].length;i++){
                                                    if (column_selected.includes("coordinates")) {
                                                        if (df0[column_selected][i]){
                                                            if (key == "y"){
                                                                y_val = y_val - 1
                                                                df[key].push(y_val);
                                                            } else {
                                                                df[key].push(df0[key][i]); 
                                                            }
                                                        }
                                                    } else if (df0[column_selected][i] & (color_selected>0.5) & (df0[column_selected.replace("_check", "_suggestion")][i]=="")) {
                                                        if (key == "y"){
                                                            y_val = y_val - 1
                                                            df[key].push(y_val);
                                                        } else {
                                                            df[key].push(df0[key][i]); 
                                                        }
                                                    } else if (df0[column_selected][i] & (color_selected<=0.5) & (df0[column_selected.replace("_check", "_suggestion")][i]!="")) {
                                                        if (key == "y"){
                                                            y_val = y_val - 1
                                                            df[key].push(y_val);
                                                        } else {
                                                            df[key].push(df0[key][i]); 
                                                        }
                                                    }
                                                }
                                            }
                                            
                                        }
                                        source.change.emit();
                                        legend_source.change.emit();
                                    }
                                    console.log("Ended")
                                    cb_obj.indices = [];
                                    """))
        perc_plot.circle(x="x", y="y", size=9, line_width=0.5,
                         fill_color={"field": "color",
                                     "transform": CategoricalColorMapper(factors=[warning_tag, solved_tag],
                                                                         palette=["red", "orange"])},
                         fill_alpha="alpha",
                         source=legend_data)

        perc_data = np.array(perc_data)
        perc_data = ColumnDataSource(dict(
            x=perc_data[:, 0].astype(float),
            y=perc_data[:, 1].astype(float),
            perc=perc_data[:, 2]))

        image_perc = LabelSet(x="x", y="y", text="perc", source=perc_data, text_align="right", y_offset=0,
                              text_font_size="12px", text_baseline="middle")
        perc_plot.add_layout(image_perc)

        plot = column(perc_plot, row(data_table, check_plot))
        if self.latitude_tag:
            map_plot = self._create_map_plot(width + width_check, source)
            tabs = [TabPanel(child=plot, title="Details"), TabPanel(child=map_plot, title="Map")]
            plot = Tabs(tabs=tabs, tabs_location='left')

        plot = column(header, text_input, plot)

        if save_in_path:
            output_file(save_in_path, mode='inline')
            save(plot)
            os.startfile(save_in_path)
        else:
            show(plot)


@validate
def get_population_nearby(df: pd.DataFrame, radius: Union[int, float],
                          latitude_columns: str = None, longitude_columns: str = None) -> pd.DataFrame:
    min_radius = 50
    if radius < min_radius:
        raise Exception(f"Unable to find population with radius of less than {min_radius}, increase the radius.")
    population_df = get_high_resolution_population_density_df()
    df = df.rename_axis('key_mapping').reset_index()
    radius_df = __create_geo_dataframe(df, lat_tag=latitude_columns, long_tag=longitude_columns)[
        ["key_mapping", "geometry"]]
    radius_df = radius_df.to_crs({'init': 'epsg:4326'})
    radius_df["geometry"] = radius_df["geometry"].centroid
    long_tag = "Lon"
    lat_tag = "Lat"
    x_dist, y_dist = _distance_to_range_ccord(radius)
    dist = (x_dist + y_dist) / 2
    min_x, max_x = radius_df["geometry"].x.min() - x_dist, radius_df["geometry"].x.max() + x_dist
    min_y, max_y = radius_df["geometry"].y.min() - y_dist, radius_df["geometry"].y.max() + y_dist
    population_df = population_df[population_df[lat_tag].between(min_y, max_y) &
                                  population_df[long_tag].between(min_x, max_x)]
    log.info("Start Filtering population df")
    start = datetime.now()
    population_df["filter"] = False
    for index, row in radius_df.iterrows():
        pos = (population_df[lat_tag].between(row["geometry"].y - y_dist, row["geometry"].y + y_dist) &
               population_df[long_tag].between(row["geometry"].x - x_dist, row["geometry"].x + x_dist))
        population_df.loc[pos, "filter"] = True
    population_df = population_df[population_df["filter"]]
    end = datetime.now()
    log.info("Filtering population df ended in {}".format(end - start))
    log.info("Start Creating GeoDataframe")
    start = datetime.now()
    population_df = gpd.GeoDataFrame(
        population_df.drop([long_tag, lat_tag], axis=1),
        crs={'init': 'epsg:4326'},
        geometry=gpd.points_from_xy(population_df[long_tag], population_df[lat_tag]))
    end = datetime.now()
    log.info("Creating GeoDataframe ended in {}".format(end - start))
    log.info("Start Merging input data with population df")
    start = datetime.now()
    if radius_df.shape[0] > 0:
        radius_df["geometry"] = radius_df.apply(lambda x: x['geometry'].buffer(dist, cap_style=1), axis=1)
    mapping = gpd.sjoin(population_df, radius_df, op='within').groupby("key_mapping")["Population"].sum()
    end = datetime.now()
    log.info("Merging input data with population df ended in {}".format(end - start))
    df["n_residents"] = df["key_mapping"].map(mapping)
    df["n_residents"].fillna(0, inplace=True)
    df = df.drop(["key_mapping"], axis=1)
    return df


def check_locations_in_highway(df, lat_col, long_col, radius_max=50):
    highway = get_highway_shapes()
    x_dist, y_dist = _distance_to_range_ccord(radius_max)
    dist = (x_dist + y_dist) / 2
    point_df = df[[lat_col, long_col]].drop_duplicates().copy()
    point_df = gpd.GeoDataFrame(
        point_df, geometry=gpd.points_from_xy(point_df[long_col], point_df[lat_col]))
    point_df["geometry"] = point_df.apply(lambda x: x['geometry'].buffer(dist, cap_style=1), axis=1)
    point_df = gpd.tools.sjoin(point_df, highway[highway["classificazione"] == "Autostrada"], op='intersects',
                               how="left")
    point_df["in_highway"] = point_df["id"].notnull()
    point_df = point_df[[lat_col, long_col, "in_highway"]]
    df = df.merge(point_df, on=[lat_col, long_col], how="left")
    return df


def _get_nearests_highway_exits(df, lat_col, long_col, highway_exits, n=1):
    point_list = highway_exits["point"].to_list()

    def find_nearest_exit(point):
        tree = spatial.KDTree(point_list)
        _, nearest_index = tree.query([point], k=n)
        nearest_index = list(nearest_index)
        return [point_list[a] for a in nearest_index]

    df["point"] = [(x, y) for x, y in zip(df[long_col], df[lat_col])]
    df["nearest_exits"] = df["point"].apply(find_nearest_exit)
    return df


def _get_distance_to_exit_list(df):
    def find_distance_from_points(row):
        return [distance(row["point"], a).m for a in row["nearest_exits"]]

    df["distances_to_exits"] = df.apply(find_distance_from_points, axis=1)
    return df


def get_distance_to_highway(df, lat_col, long_col):
    highway_exits = get_highway_exits()
    highway_exits = highway_exits[highway_exits["classificazione"] == "Autostrada"]
    highway_exits["point"] = [(x, y) for x, y in zip(highway_exits["geometry"].x, highway_exits["geometry"].y)]
    df = _get_nearests_highway_exits(df, lat_col, long_col, highway_exits, n=1)
    df = _get_distance_to_exit_list(df)
    df["distance_from_highway"] = df["distances_to_exits"].str[0]
    del df["nearest_exits"], df["distances_to_exits"]
    return df

# class KDEDensity:
#
#     def __init__(self, df_density, lat_tag, long_tag, value_tag=None):
#         self.df_density = df_density
#         self.lat_tag = lat_tag
#         self.long_tag = long_tag
#         self.value_tag = value_tag
#         self.kde = None
#         self.run_kde()
#
#     def run_kde(self):
#         Xtrain = np.vstack([self.df_density[self.lat_tag],
#                             self.df_density[self.long_tag]]).T
#         #Xtrain *= np.pi / 180.
#
#         self.kde = KernelDensity(bandwidth=0.05, metric='haversine',
#                             kernel='gaussian', algorithm='ball_tree')
#
#         if self.value_tag is not None:
#             Ytrain = self.df_density[self.value_tag].values.T
#             Ytrain[Ytrain <= 0] = 0.0001
#             self.kde.fit(Xtrain, sample_weight=Ytrain)
#         else:
#             self.kde.fit(Xtrain)
#
#     def evaluate_in_point(self, lat, long):
#         xy = np.vstack([[lat], [long]]).T
#         #xy *= np.pi / 180.
#         Z = np.exp(self.kde.score_samples(xy))
#         return Z[0]
