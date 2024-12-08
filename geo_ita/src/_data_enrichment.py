import difflib
import logging
import ssl
from datetime import datetime
from typing import Dict, Optional
import requests

from valdec.decorators import validate
from bs4 import BeautifulSoup
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import geopandas as gpd
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from googlesearch import search
from googleapiclient.discovery import build

from geo_ita.src.utils import *
import geo_ita.src.config as cfg
from geo_ita.src._data import (
    get_df, get_df_comuni, get_administrative_changes_df, _clean_denomination_text,
    get_double_languages_denomination,
    _get_shape_italia, get_high_resolution_population_density_df
)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx


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


class AddGeographicalInfo:
    """
    This class enriches a DataFrame with geographical information based on Italian administrative divisions.
    
    It matches the input data with ISTAT registry data to add details such as comune, provincia, regione, 
    and other geographical attributes. The class provides methods for setting tags, running matches, 
    and retrieving unmatched values.
    """
    MATCH_COLUMN = "geo_ita_match_column"
    SUFFIX_DEFAULT = "_geo_ita_suffix_default"
    OUTPUT_COLUMNS = [
        cfg.TAG_COMUNE, cfg.TAG_CODICE_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
        cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE, cfg.TAG_AREA_GEOGRAFICA, cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE
    ]

    @validate
    def __init__(self, df: pd.DataFrame):
        # Initialize with the original DataFrame and set default attributes
        self.original_df = df.where(pd.notnull(df), None)
        self.detail_level = {}
        self.df = None
        self.istat_registry = None
        self.not_match = None

    @validate
    def set_comuni_tag(self, column_name: str):
        """
        Set the column name for the 'comuni' tag in the dataframe.

        Args:
            column_name (str): The name of the column to be used for the 'comuni' tag.
        """
        self._set_tag(column_name, GeoLevel.COMUNE)

    @validate
    def set_province_tag(self, column_name: str):
        """
        Set the column name for the 'province' tag in the dataframe.

        Args:
            column_name (str): The name of the column to be used for the 'province' tag.
        """
        self._set_tag(column_name, GeoLevel.PROVINCIA)

    @validate
    def set_regioni_tag(self, column_name: str):
        """
        Set the column name for the 'regioni' tag in the dataframe.

        Args:
            column_name (str): The name of the column to be used for the 'regioni' tag.
        """
        self._set_tag(column_name, GeoLevel.REGIONE)

    def _set_tag(self, column_name: str, geo_level: GeoLevel):
        # Validate and set the tag for the specified geographical level
        test_column_in_dataframe(self.original_df, column_name)
        code_level = infer_geographical_category(list(self.original_df[column_name].unique()))
        if code_level == CodeLevel.SIGLA and geo_level != GeoLevel.PROVINCIA:
            raise ValueError(f"The column cannot be of type SIGLA for {geo_level.name.lower()}.")
        self.detail_level[geo_level] = (column_name, code_level)

    def run_simple_match(self):
        """
        Perform a simple match of the dataframe against the ISTAT registry based on the set detail levels.
        """
        if not self.detail_level:
            raise ValueError("No detail level set for matching.")

        self._prepare_dataframe_for_matching()
        self._match_with_istat_registry()

    def _prepare_dataframe_for_matching(self):
        """
        Prepare the dataframe for matching by sorting detail levels, 
        selecting relevant columns, and creating a match column.
        """
        self.detail_level = dict(sorted(self.detail_level.items()))
        geo_columns = [name for name, _ in self.detail_level.values()]
        self.df = self.original_df[geo_columns].copy().drop_duplicates()
        self.df[self.MATCH_COLUMN] = self.df[geo_columns[0]]
        self.df = self.df[self.df[self.MATCH_COLUMN].notnull()]

    def _match_with_istat_registry(self):
        # Match the dataframe with the ISTAT registry based on the detail levels
        geo_level = list(self.detail_level.keys())[0]
        geo_code = list(self.detail_level.values())[0][1]
        self.istat_registry = get_df(geo_level)
        geo_registry_tag = get_tag_registry(geo_code, geo_level)
        self.istat_registry[self.MATCH_COLUMN] = self.istat_registry[geo_registry_tag]
        if geo_code == CodeLevel.SIGLA:
            self._run_sigla_match()
        elif geo_code == CodeLevel.CODE:
            self._run_code_match()
        else:
            self._run_denomination_match()

    def _check_non_match(self, istat_values):
        # Check for non-matching values in the dataframe
        self.not_match = self.df[(~self.df[self.MATCH_COLUMN].isin(istat_values))]
        return self.not_match[self.MATCH_COLUMN].nunique()

    def _run_sigla_match(self):
        # Perform matching based on 'sigla' (abbreviation)
        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].str.lower().str.strip()
        self.istat_registry[self.MATCH_COLUMN] = self.istat_registry[self.MATCH_COLUMN].str.lower()

        istat_values = list(self.istat_registry[self.MATCH_COLUMN].unique())

        n_not_match = self._check_non_match(istat_values)
        n_tot = self.df[self.MATCH_COLUMN].nunique()
        if n_not_match == 0:
            log.info(f"Matching completed, found {n_tot} different sigle.")
        else:
            log.warning(
                f"Matched {n_tot - n_not_match} over {n_tot}. "
                f"Missing {n_not_match} unique values ({n_not_match / n_tot:.1%}).")

    def _run_code_match(self):
        # Perform matching based on codes
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

        self.df[self.MATCH_COLUMN] = _clean_denomination_text(self.df[self.MATCH_COLUMN])
        self.istat_registry[self.MATCH_COLUMN] = _clean_denomination_text(self.istat_registry[self.MATCH_COLUMN])

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
        # Get a list of values that did not match
        if self.not_match is None:
            raise Exception("Run simple match before getting list of not matched values.")
        result = list(self.not_match[self.MATCH_COLUMN].unique())
        return result

    def _add_provincia_regione_denomination_to_not_matched(self):
        # Add province or region denomination to unmatched values
        if cfg.TAG_REGIONE + self.SUFFIX_DEFAULT in self.not_match:
            return
        if GeoLevel.PROVINCIA in self.detail_level:
            detail_column = self.detail_level[GeoLevel.PROVINCIA][0]
            addinfo = AddGeographicalInfo(self.not_match)
            addinfo.MATCH_COLUMN = addinfo.MATCH_COLUMN + "_provincia"
            addinfo.set_province_tag(detail_column)
            addinfo.run_simple_match()
            self.not_match = addinfo.get_result(suffix=self.SUFFIX_DEFAULT)
        elif GeoLevel.REGIONE in self.detail_level:
            detail_column = self.detail_level[GeoLevel.REGIONE][0]
            addinfo = AddGeographicalInfo(self.not_match)
            addinfo.MATCH_COLUMN = addinfo.MATCH_COLUMN + "_regione"
            addinfo.set_regioni_tag(detail_column)
            addinfo.run_simple_match()
            self.not_match = addinfo.get_result(suffix=self.SUFFIX_DEFAULT)

    @validate
    def get_n_not_matched(self) -> int:
        # Get the number of unmatched values
        if self.not_match is None:
            raise Exception("Run simple match before getting number of not matched values.")
        result = self.not_match[self.MATCH_COLUMN].nunique()
        return result

    def _test_if_df_contains_homonym_comuni(self):
        # Test if the dataframe contains homonym comuni (municipalities with the same name)
        comuni_homonym_df = self._calculate_italian_comuni_homonym()
        # Check if the dataset contains homonym comune
        homonym_comuni_list = list(set(comuni_homonym_df[cfg.TAG_COMUNE]))
        match_homonym_comuni = self.df[self.MATCH_COLUMN].isin([a.lower() for a in homonym_comuni_list])
        if not match_homonym_comuni.any():
            return

        log.info(f"Found homonym comuni on dataset.")
        if GeoLevel.PROVINCIA in self.detail_level:
            geo_code = self.detail_level[GeoLevel.PROVINCIA][1]
            detail_column = self.detail_level[GeoLevel.PROVINCIA][0]
            registry_column_detail = get_tag_registry(geo_code, GeoLevel.PROVINCIA)
        elif GeoLevel.REGIONE in self.detail_level:
            geo_code = self.detail_level[GeoLevel.REGIONE][1]
            detail_column = self.detail_level[GeoLevel.REGIONE][0]
            registry_column_detail = get_tag_registry(geo_code, GeoLevel.REGIONE)
        else:
            log.warning(
                "You can distinguish them only by using another geographic information (ex.: provincia or regione). "
                "If you want to identify the right comune add provincia or regione detail or homonym comuni will be "
                "ignored.")
            self.df.loc[match_homonym_comuni, self.MATCH_COLUMN] = ""
            #registry_column_detail = get_tag_registry(CodeLevel.SIGLA, GeoLevel.PROVINCIA)
            #comuni_homonym_df["key"] = comuni_homonym_df[cfg.TAG_COMUNE] + " " + comuni_homonym_df[
            #    registry_column_detail]
            #self.istat_registry = self._split_comuni_homonym(self.istat_registry, registry_column_detail,
            #                                                 comuni_homonym_df)
            return
        log.info(f"The column {detail_column} will be used in order to found the right comune.")
        comuni_homonym_df["key"] = comuni_homonym_df[cfg.TAG_COMUNE] + " " + comuni_homonym_df[registry_column_detail]
        self.df = self._split_comuni_homonym(self.df, detail_column, comuni_homonym_df)
        self.istat_registry = self._split_comuni_homonym(self.istat_registry, registry_column_detail, comuni_homonym_df)

    def _calculate_italian_comuni_homonym(self):
        # Calculate homonym comuni in Italy
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
        # Split homonym comuni in the dataframe
        denomination_column = AddGeographicalInfo.MATCH_COLUMN

        pos = df[denomination_column].isin(comuni_homonym_df[cfg.TAG_COMUNE].unique())
        df[denomination_column] = df[denomination_column].where(~pos,
                                                                df[denomination_column] + " " + df[
                                                                    details_columns].astype(
                                                                    str).str.lower())
        df[denomination_column] = df[denomination_column].replace(comuni_homonym_df.set_index("key")["new_name"])
        return df

    def _find_any_bilingual_name(self):
        # Find and replace bilingual names in the dataframe
        geo_level = list(self.detail_level.keys())[0]
        replace_multilanguage_name = get_double_languages_denomination(geo_level, self.istat_registry)

        for k, v in replace_multilanguage_name.items():
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(k, v)
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(f"{k} {v}", v)
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(f"{v} {k}", v)

    def _rename_any_english_name(self):
        # Rename any English names in the dataframe
        s: pd.Series = self.df[self.MATCH_COLUMN].replace(cfg.rename_english_name)
        match_english_name = s != self.df[self.MATCH_COLUMN]
        if match_english_name.any():
            replaces = self.df[match_english_name][self.MATCH_COLUMN].unique()
            log.info(f"Replaced {len(replaces)} name written in english: {replaces}")
            self.df[self.MATCH_COLUMN] = s

    def _find_any_variation_from_istat_history(self, df_changes=None, i=0):
        # Find any variations from ISTAT history
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
            df_changes[cfg.TAG_COMUNE] = _clean_denomination_text(df_changes[cfg.TAG_COMUNE])
            df_changes["new_denominazione_comune"] = _clean_denomination_text(df_changes["new_denominazione_comune"])
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
        # Try custom replacements for denominations
        list_den_not_found = self.get_not_matched_list()
        dict_den_anag = {self._custom_replace_denomination(a): a for a in istat_values}
        dict_den_not_found = {a: self._custom_replace_denomination(a) for a in list_den_not_found}
        dict_den_not_found = {k: dict_den_anag[v] for k, v in dict_den_not_found.items() if v in dict_den_anag}
        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(dict_den_not_found)

    @staticmethod
    def _custom_replace_denomination(value):
        # Custom replacement for denomination values
        for v in cfg.clear_den_replace:
            value = value.replace(v[0], v[1])
        value = " ".join(value.split())
        return value

    @staticmethod
    def _check_if_text_is_comune(value):
        # Check if the text is a valid comune
        info = get_geo_info_from_comune(comune=value, flag_find_frazioni=False)
        if info is None:
            return None
        else:
            return info[cfg.TAG_COMUNE]

    def _get_info_from_address(self, address):
        # Extract information from an address
        match = re.search(
            r'(?P<comune2>[^,]+, )?(?P<comune1>[^,]+), (?P<provincia>[^,]+), (?P<regione>[^,0-9]+)(?P<cap>, [0-9]{5})?, Italia',
            address)
        if not match:
            return None
        comune = match.group("comune1")
        comune = self._check_if_text_is_comune(comune)
        if comune is None:
            comune = match.group("comune2")
            if comune is not None:
                comune = comune[:-2]
                comune = self._check_if_text_is_comune(comune)
        if comune is None:
            comune = match.group("provincia")
            comune = comune.replace("Roma Capitale", "Roma")
            comune = self._check_if_text_is_comune(comune)
        if comune is not None:
            comune = clean_denomination_text_value(comune)
        return comune

    def _check_matched_comune(self, row, value, match_dict):
        # Check if the matched value is a valid comune
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
        # Run the process to find frazioni (subdivisions) using Nominatim
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
            if (row[self.MATCH_COLUMN] is None) or (row[self.MATCH_COLUMN] == ""):
                pass
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
        # Run a query using Google search
        try:
            # First Try
            return list(
                search(query, tld='com', num=n_url_read, lang="it", country="Italy", stop=n_url_read, pause=2.5,
                       verify_ssl=False))
        except Exception as e:
            log.error('Failed to search on google tentative 1: ' + str(e))
            try:
                results = google_query(query, cfg.google_search_api_key, cfg.google_search_cse_id, num=n_url_read)
                return [result['link'] for result in results]
            except Exception as e:
                log.error('Failed to search on google tentative 2: ' + str(e))
                return []

    def _find_info_on_page(self, url, denomination):
        # Find information on a webpage
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, verify=False)
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
        sentences = clean_htmltext(sentences)
        sentences = re.split(r'[\r\n\.]', sentences)
        matches = []
        for text in sentences:
            results = re.findall(cfg.regex_find_frazioni.format(denomination), text)
            for result in results:
                _match = result[8].split("provincia")[0]
                _match = [clean_denomination_text_value(comune) for comune in
                          self.istat_registry[cfg.TAG_COMUNE].unique() if
                          re.match(f"\\b{comune.lower()}\\b", _match)]
                matches.extend(_match)
        matches = list(set(matches))
        return matches

    @validate
    def run_find_frazioni_on_web(self, n_url_read: int = 1):
        # Run the process to find frazioni (subdivisions) using web search
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
            if (row[self.MATCH_COLUMN] is None) or (row[self.MATCH_COLUMN] == ""):
                pass
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
            log.info(
                f"Match {len(match_dict)} name that corresponds to a possible frazione of a comune from Web:\n{match_dict}")
            _ = self._check_non_match(self.istat_registry[self.MATCH_COLUMN].unique())

    @validate
    def get_result(
        self,
        add_missing: bool = False,
        drop_not_match: bool = False,
        suffix: Optional[str] = None,
        handle_duplicate_column: str = "error"
    ) -> pd.DataFrame:
        """
        Get the result dataframe after matching.

        Args:
            add_missing (bool): Whether to add missing values to the result.
            drop_not_match (bool): Whether to drop non-matching values from the result.
            suffix (str): The suffix to add to the result columns.
            handle_duplicate_column (str): How to handle duplicate columns in the result.
            Values can be 'error', 'overwrite', or 'progressive'.
        """
        # Get the result dataframe after matching
        if self.not_match is None:
            raise Exception("Run simple match before get the result.")

        # Log results
        n_not_match = self.not_match[self.MATCH_COLUMN].nunique()
        if n_not_match > 0:
            log.warning(f"Unable to find {n_not_match} unique values: {self.get_not_matched_list()}")
        else:
            log.info(f"Found every values.")

        join_columns = [name for name, code in self.detail_level.values()]
        original_index = self.original_df.index
        self.original_df = self.original_df.merge(self.df[[self.MATCH_COLUMN] + join_columns], on=join_columns,
                                                  how="left")
        self.original_df.index = original_index

        # Check column names in original dataset for any duplicates
        output_columns = list(set(self.OUTPUT_COLUMNS).intersection(self.istat_registry.columns))
        self.istat_registry = self.istat_registry[[self.MATCH_COLUMN] + output_columns]

        check_duplicate_column_output(self.original_df, self.istat_registry, output_columns,
                                      suffix,  handle_duplicate_column, log)
        if add_missing:
            how = "outer" if not drop_not_match else "left"
        else:
            how = "inner" if drop_not_match else "left"

        result = (
            self.original_df
            .merge(self.istat_registry, on=self.MATCH_COLUMN, how=how)
        )
        if how == "left":
            self.original_df.index = original_index

        result.drop(columns=[self.MATCH_COLUMN], inplace=True)
        return result

    @validate
    def run_similarity_match(self, unique_flag: bool = False, threshold=cfg.min_acceptable_similarity):
        # Run similarity matching for unmatched values
        if unique_flag:
            input_den = self.df[self.MATCH_COLUMN].values()
            registry_not_matched = [a for a in self.istat_registry[self.MATCH_COLUMN] if a not in input_den]
            match_dict = self._find_match(registry_not_matched, self.get_not_matched_list(), unique=True,
                                          threshold=threshold)
            self.similarity_result = {v[0]: (k, v[1]) for k, v in match_dict.items()}
        else:
            self.similarity_result = self._find_match(self.get_not_matched_list(),
                                                      self.istat_registry[self.MATCH_COLUMN],
                                                      threshold=threshold)
        n = len(self.similarity_result)
        if n > 1:
            log.info(f"Match {n} name by similarity:\n{self.similarity_result}")
        else:
            log.info("No match by similarity")

    def get_similarity_result(self):
        # Get the result of similarity matching
        return self.similarity_result

    def accept_similarity_result(self):
        # Accept the similarity matching results
        if self.similarity_result is not None:
            self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(
                {k: v[0] for k, v in self.similarity_result.items()})
            _ = self._check_non_match(self.istat_registry[self.MATCH_COLUMN])
        else:
            raise Exception("Run 'run_similarity_match' before accept_similarity_result.")

    @validate
    def use_manual_match(self, manual_dict: Dict[str, str]):
        # Use manual matching for unmatched values
        geo_level = list(self.detail_level.keys())[0]
        geo_code = list(self.detail_level.values())[0][1]

        # Checks
        if (geo_code != CodeLevel.DENOMINATION) or (geo_level != GeoLevel.COMUNE):
            raise Exception("Run use_manual_match only to find comune by denomination.")

        self.df[self.MATCH_COLUMN] = self.df[self.MATCH_COLUMN].replace(manual_dict)
        _ = self._check_non_match(self.istat_registry[self.MATCH_COLUMN])

    @staticmethod
    def _find_match(not_match1, not_match2, threshold, unique=False) -> Dict:
        """
        Find the best match for each value in not_match1 from not_match2

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


def _create_geo_dataframe(df0, lat_tag=None, long_tag=None, geo_tag=None):
    if isinstance(df0, gpd.GeoDataFrame):
        df = df0.copy()
        if df.crs is None:
            coord_system = __find_coordinates_system(df, geometry="geometry")
            df.crs = {'init': coord_system}
    elif isinstance(df0, pd.DataFrame):
        if lat_tag is not None and long_tag is not None:
            df = df0[df0[long_tag].notnull() & df0[lat_tag].notnull()]
            df = gpd.GeoDataFrame(
                df.drop([long_tag, lat_tag], axis=1), geometry=gpd.points_from_xy(df[long_tag], df[lat_tag]))
            coord_system = __find_coordinates_system(df, lat_tag, long_tag)
            df.crs = {'init': coord_system}
        elif geo_tag is not None:
            df = gpd.GeoDataFrame(df0, geometry=geo_tag)
            coord_system = __find_coordinates_system(df0, geometry=geo_tag)
            df.crs = {'init': coord_system}
        elif "geometry" in df0.columns:
            df = gpd.GeoDataFrame(df0)
            coord_system = __find_coordinates_system(df0, geometry="geometry")
            df.crs = {'init': coord_system}
            log.info("Found geometry columns")
        else:
            flag_coord_found, lat_tag, long_tag = __find_coord_columns(df0)
            if not flag_coord_found:
                raise Exception("The DataFrame must have a geometry attribute or lat-long.")
            df = df0[df0[long_tag].notnull() & df0[lat_tag].notnull()]
            df = gpd.GeoDataFrame(
                df.drop([long_tag, lat_tag], axis=1), geometry=gpd.points_from_xy(df[long_tag], df[lat_tag]))
            coord_system = __find_coordinates_system(df, lat_tag, long_tag)
            df.crs = {'init': coord_system}
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
def get_geo_info_from_comune(comune: str, provincia: str = None, regione: str = None,
                             flag_find_frazioni: bool = True) -> Dict[str, str]:
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
def get_city_from_coordinates(
    df: pd.DataFrame,
    latitude_column: Optional[str] = None,
    longitude_column: Optional[str] = None,
    geometry_column: Optional[str] = None,
    suffix: Optional[str] = None,
    handle_duplicate_column: str = "error"
) -> pd.DataFrame:
    """
    Map geographic information (city, province, region) to a dataframe based on coordinates.

    Args:
        df (pd.DataFrame): Input dataframe containing coordinate columns.
        latitude_column (str, optional): Name of the column containing latitude values.
        longitude_column (str, optional): Name of the column containing longitude values.
        geometry_column (str, optional): Name of the column containing geometry values.
        suffix (str): the suffix of result columns.
        handle_duplicate_column (str): How to handle duplicate columns in the result.
            Possible values are 'error', 'overwrite', or 'progressive'.
    Returns:
        pd.DataFrame: Input dataframe enriched with geographic information.
    """
    # Validate the presence of latitude and longitude columns
    if latitude_column:
        test_column_in_dataframe(df, latitude_column)
    if longitude_column:
        test_column_in_dataframe(df, longitude_column)
    if geometry_column:
        test_column_in_dataframe(df, geometry_column)

    # Add a unique key to map results back to the original dataframe
    df["key_mapping"] = range(len(df))

    # Load official geographic data
    df_comuni = get_df_comuni()
    df_comuni = gpd.GeoDataFrame(df_comuni)
    df_comuni.crs = "epsg:32632"  # Original CRS (UTM)
    df_comuni = df_comuni.to_crs("epsg:4326")  # Convert to WGS84

    # Create a GeoDataFrame from the input dataframe coordinates
    geo_df = _create_geo_dataframe(df, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)
    geo_df = geo_df[geo_df["geometry"].notnull()]  # Drop rows with missing geometries
    geo_df = geo_df[["key_mapping", "geometry"]].drop_duplicates()  # Remove duplicate geometries

    # Ensure points are in the correct projection
    geo_df["geometry"] = geo_df["geometry"].centroid
    geo_df["prova_x"] = geo_df["geometry"].x
    geo_df["prova_y"] = geo_df["geometry"].y
    geo_df = geo_df.to_crs("epsg:4326")

    # Perform spatial join with city boundaries
    map_city = gpd.sjoin(geo_df, df_comuni, op="within", how="left")

    # Log missing points
    missing_points = map_city[map_city[cfg.TAG_COMUNE].isna() & (~map_city["geometry"].is_empty)]["geometry"].unique()
    if missing_points:
        log.warning(f"Unable to find the city for {len(missing_points)} points: "
                    f"{[(pt.x, pt.y) for pt in missing_points]}")
    else:
        log.info("Found the correct city for each point.")

    # Select relevant columns for the final mapping
    output_column = [cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]
    map_city = map_city[["key_mapping"] +output_column]

    check_duplicate_column_output(df, map_city, output_column, suffix, handle_duplicate_column, log)

    # Merge the results back to the original dataframe
    original_index = df.index
    result = (
        df
        .merge(map_city, on="key_mapping", how="left")
        .drop(columns=["key_mapping"])
    )
    result.index = original_index

    # Clean up temporary columns from the original dataframe
    df.drop(columns=["key_mapping"], inplace=True, errors="ignore")

    return result


def __test_city_in_address(df, city_tag, address_tag):
    return df.apply(lambda x: x[city_tag].lower() in x[address_tag]
    if (x[address_tag] and x[city_tag])
    else False,
                    axis=1)


def _fetch_urls(address, n_url_read):
    """
    Fetch URLs for an address from Google search.

    Args:
        address (str): The address to search for.
        n_url_read (int): The number of URLs to retrieve.

    Returns:
        list: A list of URLs where the address was found.
    """
    urls = []
    try:
        # Attempt to search via Google
        urls = list(search(address, tld='com', num=n_url_read, lang="it", country="Italy", stop=n_url_read, pause=2.5,
                           verify_ssl=False))
    except Exception as e:
        log.error(f'Failed to search on Google (attempt 1): {str(e)}')
        try:
            # Fallback to Google Search API if the first attempt fails
            my_results = google_query(address, cfg.google_search_api_key, cfg.google_search_cse_id, num=n_url_read)
            urls.extend(result['link'] for result in my_results)
        except Exception as e:
            log.error(f'Failed to search on Google (attempt 2): {str(e)}')
    return urls


def _scrape_url_for_address(url, pattern):
    """
    Scrape a URL for the pattern corresponding to an address.

    Args:
        url (str): The URL to scrape.
        pattern (str): The regex pattern to search for.

    Returns:
        str or None: The matching address if found, otherwise None.
    """
    try:
        res = requests.get(url, verify=False)
        soup = BeautifulSoup(res.content, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        text = clean_htmltext(text)  # Clean up the extracted text

        # Search for the pattern in the page's text
        match = re.search(pattern, text)
        return match.group() if match else None
    except requests.RequestException as e:
        log.error(f"Failed to fetch or scrape URL {url}: {str(e)}")
    except Exception as e:
        log.error(f"An error occurred while processing URL {url}: {str(e)}")
    return None


def _try_replace_abbreviation_on_google(df, n_url_read, geocode):
    """
        Attempt to find and correct abbreviated street names using Google search.

        Args:
            df (DataFrame): The dataframe containing the addresses to process.
            n_url_read (int): The number of URLs to read for each address.
            geocode (function): A geocoding function to retrieve coordinates.

        Returns:
            DataFrame: The updated dataframe with the resolved addresses.
        """
    log.info("Trying to find abbreviated names")

    # Collect addresses that haven't been geocoded yet
    not_found = list(df.loc[df["location"].isna(), "address_search"].unique())

    # Process each address that hasn't been found
    for address in not_found:
        address_without_city = address.split(",")[0]
        # Regex to extract the street name and abbreviation
        m = re.search(r"^([^.]+) (([a-z]+\. ?)+) ?([^.]+)$", address_without_city)
        if m:
            prefix = m.group(1)
            suffix = m.group(4)
            abbreviations = m.group(2).replace(" ", "")  # Remove spaces in abbreviation

            # Build a regex for abbreviations to match similar patterns
            abbreviations_pattern = "[a-z]+ ?".join(abbreviations.split("."))
            pattern = f"{prefix} ?{abbreviations_pattern} ?{suffix}"

            # Initialize match list and URLs
            match = []
            urls = _fetch_urls(address, n_url_read)

            # If URLs were found, attempt to find the address in the web content
            if urls:
                for url in urls:
                    found_match = _scrape_url_for_address(url, pattern)
                    if found_match:
                        match.append(found_match)

            match = list(set(match))  # Remove duplicates

            # If a unique match was found, update the address and geocode
            if len(match) == 1:
                pos_match = df["address_search"] == address
                df.loc[pos_match, "address_search"] = match[0]

                # Rebuild the address for geocoding and fetch coordinates
                full_address = f"{match[0]}, {', '.join(address.split(',')[1:])}" if "," in address else match[0]
                location = geocode(full_address)

                if location:
                    df.loc[pos_match, "latitude"] = location.latitude
                    df.loc[pos_match, "longitude"] = location.longitude
                    df.loc[pos_match, "address_test"] = location.address.lower()

    return df


def _try_wrong_replace_of_apostrophe(df, address_tag, geocode):
    """
    Attempt to correct addresses where apostrophes might have been wrongly replaced.

    This function searches for common address prefixes that may have had their apostrophes
    incorrectly removed (e.g., 'del' -> 'del', 'dell' -> 'dell', etc.) and restores them.

    Args:
        df (DataFrame): The DataFrame containing addresses to process.
        address_tag (str): The name of the column containing address strings.
        geocode (function): A geocoding function to retrieve coordinates.

    Returns:
        DataFrame: The updated DataFrame with corrected addresses and coordinates.
    """
    log.info("Attempting to find and fix incorrect apostrophe replacements in addresses.")

    # Define the regex pattern for identifying address prefixes that may be missing an apostrophe.
    regex = r"\b(del|dell|d|nell|sull|sant|Sant)([A-Z][^\s]+)"

    # Find rows where the address contains a prefix that may need apostrophe correction
    pos_replace = (df["location"].isna() & df[address_tag].str.contains(regex))
    addresses_to_fix = df.loc[pos_replace, address_tag].unique()

    # Iterate over addresses that need fixing
    for address in addresses_to_fix:
        # Apply the regex to restore the apostrophe
        new_name = re.sub(regex, r"\1'\2", address)

        # Find the positions of the current address in the DataFrame
        pos_match = df[address_tag] == address

        # Update the address to the corrected version
        df.loc[pos_match, "address_search"] = new_name

        # Try to geocode the corrected address
        location = geocode(new_name)
        if location:
            # Update latitude, longitude, and address_test if location is found
            df.loc[pos_match, "latitude"] = location.latitude
            df.loc[pos_match, "longitude"] = location.longitude
            df.loc[pos_match, "address_test"] = location.address.lower()
            log.info(f"Geocoded address: {new_name} -> {location.latitude}, {location.longitude}")
        else:
            log.warning(f"Could not geocode address: {new_name}")
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
def get_coordinates_from_address(
    df: pd.DataFrame,
    address_column: str,
    comune_column: Optional[str] = None,
    provincia_column: Optional[str] = None,
    regione_column: Optional[str] = None,
    n_url_read: int = 1,
    suffix: Optional[str] = None,
    handle_duplicate_column: str = "error"
) -> pd.DataFrame:
    """
    Finds coordinates for addresses in a DataFrame using OpenStreetMap data.

    Args:
        df (pd.DataFrame): Input DataFrame with address information.
        address_column (str): Column containing the address.
        comune_column (str, optional): Column with comuni names.
        provincia_column (str, optional): Column with province names.
        regione_column (str, optional): Column with regioni names.
        n_url_read (int): Number of retries for external API requests.
        suffix (str): the suffix of result columns.
        handle_duplicate_column (str): How to handle duplicate columns in the result.
            Possible values are 'error', 'overwrite', or 'progressive'.

    Returns:
        pd.DataFrame: Original DataFrame enriched with coordinates.
    """
    # Validate input columns
    test_column_in_dataframe(df, address_column)
    for tag in [comune_column, provincia_column, regione_column]:
        if tag:
            test_column_in_dataframe(df, tag)

    # Prepare a unique subset of data for processing
    relevant_columns = [col for col in [address_column, comune_column, provincia_column, regione_column] if col]
    unique_addresses = df[relevant_columns].drop_duplicates()
    unique_addresses["address_search"] = unique_addresses[address_column].str.lower()

    # Enrich the address with municipality if not already included
    if comune_column:
        condition = __test_city_in_address(unique_addresses, comune_column, "address_search") | unique_addresses[
            comune_column].isna()
        unique_addresses["address_search"] = np.where(
            condition,
            unique_addresses["address_search"],
            unique_addresses["address_search"] + ", " + unique_addresses[comune_column].str.lower()
        )

    # Initialize geolocator and geocode function
    geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Attempt to find locations using OpenStreetMap
    unique_addresses = _find_location_with_openstreetmap(unique_addresses, geocode)

    # Handle cases where locations were not found
    if unique_addresses["location"].isna().sum() > 0:
        unique_addresses = _try_replace_abbreviation_on_google(unique_addresses, n_url_read, geocode)
        unique_addresses = _try_wrong_replace_of_apostrophe(unique_addresses, address_column, geocode)

    # Validate addresses against municipality, province, and region
    unique_addresses = _test_address_with_comune_provincia_regione(
        unique_addresses, comune_column, provincia_column, regione_column
    )

    # Clean up intermediate columns
    unique_addresses.drop(columns=["address_search", "location", "address_test", "test"], errors="ignore",
                          inplace=True)
    check_duplicate_column_output(df, unique_addresses, ["latitude", "longitude"], suffix, handle_duplicate_column, log)

    # Merge results back into the original DataFrame
    result_df = df.merge(unique_addresses, how="left", on=relevant_columns)
    return result_df


@validate
def get_address_from_coordinates(
    df: pd.DataFrame,
    latitude_column: Optional[str] = None,
    longitude_column: Optional[str] = None,
    geometry_column: Optional[str] = None,
    suffix: Optional[str] = None,
    handle_duplicate_column: str = "error"
) -> pd.DataFrame:
    """
    Retrieve addresses from latitude and longitude coordinates.

    Args:
        df (pd.DataFrame): Input DataFrame containing coordinates.
        latitude_column (str): Column name for latitude. Automatically detected if not provided.
        longitude_column (str): Column name for longitude. Automatically detected if not provided.
        geometry_column (str): Column name for geometry. Automatically detected if not provided.
        suffix (str): the suffix of result columns.
        handle_duplicate_column (str): How to handle duplicate columns in the result.
            Possible values are 'error', 'overwrite', or 'progressive'.

    Returns:
        pd.DataFrame: DataFrame with addresses and cities extracted from coordinates.
    """
    df = _create_geo_dataframe(df, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)
    df.to_crs("epsg:4326", inplace=True)

    # Prepare unique coordinate pairs
    coordinates_df = df[["geometry"]].drop_duplicates()
    coordinates_df["coordinates"] = (
            coordinates_df["geometry"].y.map(str) + ", " + coordinates_df["geometry"].x.map(str)
    )

    # Estimate the time required
    num_coordinates = coordinates_df.shape[0]
    log.info(f"Processing {num_coordinates} coordinate pairs. Estimated time: at least {num_coordinates} seconds.")

    # Initialize geolocator with a rate limiter
    geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)
    reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    # Perform reverse geocoding
    start_time = datetime.now()
    coordinates_df["location"] = coordinates_df["coordinates"].apply(reverse_geocode)
    log.info(f"Reverse geocoding completed in {datetime.now() - start_time} seconds.")

    # Define address and city column names, avoiding collisions
    address_col = "address"
    city_col = "city"

    # Extract and clean address and city information
    coordinates_df[address_col] = coordinates_df["location"].apply(
        lambda loc: loc.address.lower() if loc else None
    )
    coordinates_df[city_col] = coordinates_df["location"].apply(
        lambda loc: loc.raw["address"].get("city") if loc and "address" in loc.raw and "city" in loc.raw[
            "address"] else None
    )

    # Drop temporary columns
    coordinates_df.drop(columns=["coordinates", "location"], inplace=True)

    check_duplicate_column_output(df, coordinates_df, [address_col, city_col], suffix, handle_duplicate_column, log)

    # Merge back with the original DataFrame
    df = df.merge(coordinates_df, how="left", on=["geometry"])
    df = pd.DataFrame(df.drop(columns=["geometry"]))

    return df


@validate
def aggregate_point_by_distance(
    df: pd.DataFrame,
    distance_in_meters: Union[int, float],
    latitude_column: str = None,
    longitude_column: str = None,
    geometry_column: str = None,
    output_column: str = "aggregation_code"
) -> pd.DataFrame:
    """
    Aggregates points into clusters based on a specified distance, associating points to a cluster based on their proximity.

    Args:
        df (pd.DataFrame): Input DataFrame containing latitude and longitude columns.
        distance_in_meters (Union[int, float]): The maximum distance between points to consider them in the same cluster.
        latitude_column (str): Column name for latitude. Optional if inferred.
        longitude_column (str): Column name for longitude. Optional if inferred.
        geometry_column (str): Column name for geometry. Optional if inferred.
        output_column (str): Column name for the aggregation cluster ID.

    Returns:
        pd.DataFrame: DataFrame with an additional column indicating the aggregation cluster.
    """
    # Validate and prepare coordinate columns
    if latitude_column is not None:
        test_column_in_dataframe(df, latitude_column)
    if longitude_column is not None:
        test_column_in_dataframe(df, longitude_column)

    if output_column in df.columns:
        log.warning(f"Column '{output_column}' already exists in the DataFrame. It will be overwritten.")

    df["key_mapping"] = range(len(df))

    # Create a GeoDataFrame
    gdf = _create_geo_dataframe(df, latitude_column, longitude_column, geometry_column)
    gdf = gdf.to_crs(epsg=3857)  # Project to a CRS with units in meters

    # Compute centroids (in case geometries are not points)
    gdf["geometry"] = gdf["geometry"].centroid

    # Create a buffer around each point
    buffer_gdf = gdf.copy()
    buffer_gdf["geometry"] = buffer_gdf["geometry"].buffer(distance_in_meters, cap_style=1)

    # Perform a spatial join to find points within the buffer
    joined_gdf = gpd.sjoin(gdf, buffer_gdf, op='within', how="left")
    joined_gdf = joined_gdf[["key_mapping_left", "key_mapping_right"]]

    # Create a sparse adjacency matrix for connected components
    n_points = gdf.shape[0]
    adjacency_matrix = csr_matrix(
        (np.ones(joined_gdf.shape[0]),
         (joined_gdf["key_mapping_left"].values, joined_gdf["key_mapping_right"].values)),
        shape=(n_points, n_points)
    )

    # Find connected components
    n_clusters, cluster_labels = connected_components(csgraph=adjacency_matrix, directed=False)

    # Map cluster labels back to the original DataFrame
    gdf[output_column] = cluster_labels
    df[output_column] = df["key_mapping"].map(gdf.set_index("key_mapping")[output_column])

    # Clean up and drop temporary columns
    df.drop(columns=["key_mapping"], inplace=True)

    # Logging cluster information
    largest_cluster_size = df[output_column].value_counts().max()
    log.info(f"Aggregated {df.shape[0]} points into {n_clusters} clusters. "
             f"The largest cluster contains {largest_cluster_size} points.")

    return df


@validate
def get_population_nearby(
    df: pd.DataFrame,
    radius: Union[int, float],
    latitude_column: str = None,
    longitude_column: str = None,
    geometry_column: str = None,
    output_column: str = "population"
) -> pd.DataFrame:
    """
    Calculate the total population within a specified radius for each point in the dataset.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing points with latitude and longitude.
    - radius (Union[int, float]): Radius (in meters) to search for population data.
    - latitude_column (str): Name of the latitude column in the input DataFrame.
    - longitude_column (str): Name of the longitude column in the input DataFrame.
    - geometry_column (str): Name of the geometry column in the input DataFrame.
    - output_column (str): Name of the output column containing the population data.

    Returns:
    - pd.DataFrame: DataFrame with an additional column 'n_residents' representing the population.
    """
    MIN_RADIUS = 50
    if radius < MIN_RADIUS:
        raise ValueError(f"Radius must be at least {MIN_RADIUS} meters.")

    # Load high-resolution population dataset
    population_df = get_high_resolution_population_density_df()

    # Assign unique identifiers for input DataFrame
    df["key_mapping"] = range(len(df))

    if output_column in df.columns:
        log.warning(f"Column '{output_column}' already exists in the DataFrame. It will be overwritten.")

    # Convert input DataFrame to GeoDataFrame
    points_gdf = _create_geo_dataframe(df, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)[
        ["key_mapping", "geometry"]]
    points_gdf = points_gdf.to_crs(epsg=4326)  # Convert to df_population metric projections

    # Filter population data before transformations (step 1 one big box)
    lat_margin = (2 * radius / 110540)
    lon_margin = (2 * radius / 111320)
    min_x = points_gdf["geometry"].x.min() - lon_margin
    max_x = points_gdf["geometry"].x.max() + lon_margin
    min_y = points_gdf["geometry"].y.min() - lat_margin
    max_y = points_gdf["geometry"].y.max() + lat_margin

    population_df = population_df[
        (population_df["Lon"] >= min_x) &
        (population_df["Lon"] <= max_x) &
        (population_df["Lat"] >= min_y) &
        (population_df["Lat"] <= max_y)
        ]

    # Filter population data before transformations (step 2 small boxes)
    if points_gdf.shape[0] < 10000:
        population_df["filter"] = False
        for index, row in points_gdf.iterrows():
            pos = (population_df["Lat"].between(row["geometry"].y - lat_margin, row["geometry"].y + lat_margin) &
                   population_df["Lon"].between(row["geometry"].x - lon_margin, row["geometry"].x + lon_margin))
            population_df.loc[pos, "filter"] = True
        population_df = population_df[population_df["filter"]]
        population_df.drop(columns=["filter"], inplace=True)

    points_gdf = points_gdf.to_crs(epsg=3857)  # Convert to metric projection for accurate distance calculations

    # Convert population dataset to GeoDataFrame
    log.debug("Start Creating GeoDataframe")
    start = datetime.now()
    population_df = gpd.GeoDataFrame(
        population_df.drop(["Lon", "Lat"], axis=1),
        crs="EPSG:4326",
        geometry=gpd.points_from_xy(population_df["Lon"], population_df["Lat"])
    ).to_crs(epsg=3857)
    end = datetime.now()
    log.debug(f"Creating GeoDataframe ended in {end - start}.")

    # Create buffer around each point
    points_gdf["geometry"] = points_gdf["geometry"].buffer(radius, cap_style=1)

    # Perform spatial join to aggregate population within the radius
    log.info("Start Merging input data with population df")
    population_df = gpd.sjoin(population_df, points_gdf, op="within", how="inner")
    population_df = (
        population_df.groupby("key_mapping")["Population"].sum()
    )
    end = datetime.now()
    log.info(f"Merging input data with population df ended in {end - start}")

    # Map population data back to the original DataFrame
    df[output_column] = df["key_mapping"].map(population_df).fillna(0).astype(int)

    # Drop temporary columns
    df.drop(columns=["key_mapping"], inplace=True)

    return df


def _get_margins(filter_comune=None,
                 filter_provincia=None,
                 filter_regione=None,
                 epsg=3857):
    filter_comune = ensure_list(filter_comune)
    filter_provincia = ensure_list(filter_provincia)
    filter_regione = ensure_list(filter_regione)
    if filter_comune is not None:
        filter_comune = [clean_denomination_text_value(a) for a in filter_comune]
        code = infer_geographical_category(filter_comune)
        shape = get_df(GeoLevel.COMUNE)
        tag_shape = get_tag_registry(code, GeoLevel.COMUNE)
        shape[tag_shape] = _clean_denomination_text(shape[tag_shape])
        margins = shape[shape[tag_shape].isin(filter_comune)]
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
    elif filter_provincia is not None:
        filter_provincia = [clean_denomination_text_value(a) for a in filter_provincia]
        code = infer_geographical_category(filter_provincia)
        shape = get_df(GeoLevel.PROVINCIA)
        tag_shape = get_tag_registry(code, GeoLevel.PROVINCIA)
        shape[tag_shape] = _clean_denomination_text(shape[tag_shape])
        margins = shape[shape[tag_shape].isin(filter_provincia)]
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
    elif filter_regione is not None:
        filter_regione = [clean_denomination_text_value(a) for a in filter_regione]
        code = infer_geographical_category(filter_regione)
        shape = get_df(GeoLevel.REGIONE)
        tag_shape = get_tag_registry(code, GeoLevel.REGIONE)
        shape[tag_shape] = _clean_denomination_text(shape[tag_shape])
        margins = shape[shape[tag_shape].isin(filter_regione)]
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
    else:
        margins = get_df(GeoLevel.REGIONE)
        margins["key"] = "Italia"
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
        margins = margins.dissolve(by='key')
    if len(margins) == 0:
        raise Exception("Unable to find the filter.")
    else:
        margins = margins[["geometry"]]
        margins.crs = {'init': "epsg:32632"}
        margins = margins.to_crs({'init': f'epsg:{epsg}'})
        margins_coord = margins["geometry"].values
        margins_coord = (min([margins_coord[i].bounds[0] for i in range(len(margins_coord))]),
                         min([margins_coord[i].bounds[1] for i in range(len(margins_coord))]),
                         max([margins_coord[i].bounds[2] for i in range(len(margins_coord))]),
                         max([margins_coord[i].bounds[3] for i in range(len(margins_coord))]))
        margins_coord = [[margins_coord[0], margins_coord[2]], [margins_coord[1], margins_coord[3]]]

    return margins_coord, margins
