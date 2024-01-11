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
import geo_ita.src.config as cfg
from geo_ita.src._data import (
    get_df, get_df_comuni, get_administrative_changes_df, _get_list,
    get_double_languages_mapping_comuni, get_double_languages_mapping_province,
    get_double_languages_mapping_regioni,
    _get_shape_italia, get_high_resolution_population_density_df
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
    text = re.sub('\s+', ' ', text)
    text = re.sub('[^A-Za-z0-9.]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text


@validate
def _test_column_in_dataframe(df: pd.DataFrame, col):
    if col not in df.columns:
        raise Exception("Column {} not found in DataFrame.".format(col))


class AddGeographicalInfo:
    @validate
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.where(pd.notnull(df), None)
        self.keys = None
        self.df = None
        self.comuni, self.province, self.sigle, self.regioni = None, None, None, None
        self.comuni_tag = None
        self.comuni_code = None
        self.province_tag = None
        self.province_code = None
        self.regioni_tag = None
        self.regioni_code = None
        self.level = None
        self.code = None
        self.geo_tag_input = None
        self.geo_tag_anag = None
        self.info_df = None
        self.list_anag = None
        self.not_match = None
        self.frazioni_dict = None
        self.similarity_dict = None

    @validate
    def set_comuni_tag(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        self.comuni_tag = col_name
        self.comuni_code = _code_or_desc(list(self.original_df[col_name].unique()))
        self.level = cfg.LEVEL_COMUNE
        self.geo_tag_input = col_name
        self.code = self.comuni_code

    @validate
    def set_province_tag(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        self.province_tag = col_name
        self.province_code = _code_or_desc(list(self.original_df[col_name].unique()))
        if self.level != cfg.LEVEL_COMUNE:
            self.level = cfg.LEVEL_PROVINCIA
            self.geo_tag_input = col_name
            self.code = self.province_code

    @validate
    def set_regioni_tag(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        self.regioni_tag = col_name
        self.regioni_code = _code_or_desc(list(self.original_df[col_name].unique()))
        if self.level is None:
            self.level = cfg.LEVEL_REGIONE
            self.geo_tag_input = col_name
            self.code = self.regioni_code

    def reset_tag(self):
        self.comuni_tag = None
        self.comuni_code = None
        self.province_tag = None
        self.province_code = None
        self.regioni_tag = None
        self.regioni_code = None
        self.level = None
        self.geo_tag_input = None
        self.code = None

    def run_simple_match(self):

        self.keys = [x for x in [self.comuni_tag, self.province_tag, self.regioni_tag] if x is not None]
        if len(self.keys) == 0:
            raise Exception("You need to set al least one betweeen cap_tag, comuni_tag, province_tag or regioni_tag.")
        self.df = self.original_df[self.keys].copy().drop_duplicates()

        self.df = self.df[self.df[self.geo_tag_input].notnull()]

        self.df[cfg.KEY_UNIQUE] = self.df[self.geo_tag_input]

        self.info_df = get_df(self.level)
        self.comuni, self.province, self.sigle, self.regioni = _get_list(df=self.info_df)
        self.geo_tag_anag = _get_tag_anag(self.code, self.level)
        self.info_df[cfg.KEY_UNIQUE] = self.info_df[self.geo_tag_anag]

        if self.code == cfg.CODE_SIGLA:
            if self.level != cfg.LEVEL_PROVINCIA:
                raise Exception("SIGLA ERROR")
            else:
                self._run_sigla()
        elif self.code == cfg.CODE_CODICE_ISTAT:
            self._run_codice()
        else:
            self._run_denominazione()

    def _check_non_match(self):
        self.not_match = self.df[self.df[cfg.KEY_UNIQUE].notnull() & (~self.df[cfg.KEY_UNIQUE].isin(self.list_anag))]
        return self.not_match[cfg.KEY_UNIQUE].nunique()

    def _run_sigla(self):
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.lower()
        self.info_df[cfg.KEY_UNIQUE] = self.info_df[cfg.KEY_UNIQUE].str.lower()
        self.list_anag = list(self.info_df[self.info_df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique())
        n_tot = self.df[self.df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].nunique()

        n_not_match = self._check_non_match()

        if n_not_match == 0:
            log.info("Matching completed, found {} different sigle.".format(n_tot))
        else:
            log.warning("Matched {} over {}.".format(n_tot - n_not_match, n_tot))

    def _run_codice(self):
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].astype(int, errors='ignore')
        self.info_df[cfg.KEY_UNIQUE] = self.info_df[cfg.KEY_UNIQUE].astype(int, errors='ignore')
        self.list_anag = list(self.info_df[self.info_df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique())

        n_tot = self.df[self.df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].nunique()

        n_not_match = self._check_non_match()

        if n_not_match == 0:
            log.info("Matching completed, found {} different codes.".format(n_tot))
        else:
            log.warning("Matched {} over {}.".format(n_tot - n_not_match, n_tot))

    def _run_denominazione(self):
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.lower()
        if self.level == cfg.LEVEL_COMUNE:
            self._test_if_df_contains_homonym_comuni()

        self.df[cfg.KEY_UNIQUE] = _clean_denom_text(self.df[cfg.KEY_UNIQUE])
        self.info_df[cfg.KEY_UNIQUE] = _clean_denom_text(self.info_df[cfg.KEY_UNIQUE])

        if self.level == cfg.LEVEL_COMUNE:
            self._clean_comuni_name()
        elif self.level == cfg.LEVEL_PROVINCIA:
            self._clean_province_name()

        self._find_any_bilingual_name()

        self.list_anag = list(self.info_df[self.info_df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique())

        self._rename_any_english_name()
        if self.level == cfg.LEVEL_COMUNE:
            self._find_any_variation_from_istat_history()

        n_tot = self.df[self.df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].nunique()

        _ = self._check_non_match()

        self._try_custom_replace_denom()

        n_not_match = self._check_non_match()

        if n_not_match == 0:
            log.info("Matching completed, found {} different names.".format(n_tot))
        else:
            log.warning("Matched {} over {}.".format(n_tot - n_not_match, n_tot))

    def _clean_hierarchical_structure(self):
        if self.province_tag is not None:
            self.not_match[self.province_tag + cfg.UNIQUE_TAG] = self.not_match[self.province_tag]
            if self.province_code != cfg.CODE_DENOMINAZIONE:
                tag = _get_tag_anag(self.province_code, self.province_tag)
                mapping = self.info_df[[tag, cfg.TAG_PROVINCIA]].drop_duplicates()
                if self.province_code == cfg.CODE_CODICE_ISTAT:
                    mapping[tag] = mapping[tag].astype(int, errors='ignore')
                    self.not_match[self.province_tag + cfg.UNIQUE_TAG] = self.not_match[
                        self.province_tag + cfg.UNIQUE_TAG].astype(int, errors='ignore')
                else:
                    mapping[tag] = mapping[tag].str.lower()
                    self.not_match[self.province_tag + cfg.UNIQUE_TAG] = self.not_match[
                        self.province_tag + cfg.UNIQUE_TAG].str.lower()
                mapping = mapping.set_index(tag)[cfg.TAG_PROVINCIA]
                self.not_match[self.province_tag + cfg.UNIQUE_TAG] = self.not_match[
                    self.province_tag + cfg.UNIQUE_TAG].map(mapping)
            self.not_match[self.province_tag + cfg.UNIQUE_TAG] = _clean_denom_text(
                self.not_match[self.province_tag + cfg.UNIQUE_TAG])
        if self.regioni_tag is not None:
            self.not_match[self.regioni_tag + cfg.UNIQUE_TAG] = self.not_match[self.regioni_tag]
            if self.regioni_code == cfg.CODE_DENOMINAZIONE:
                self.not_match[self.regioni_tag + cfg.UNIQUE_TAG] = _clean_denom_text(
                    self.not_match[self.regioni_tag + cfg.UNIQUE_TAG])
            else:
                tag = _get_tag_anag(self.regioni_code, self.regioni_tag)
                mapping = self.info_df[[tag, cfg.TAG_REGIONE]].drop_duplicates()
                mapping[tag] = mapping[tag].astype(int, errors='ignore')
                self.not_match[self.regioni_tag + cfg.UNIQUE_TAG] = self.not_match[
                    self.regioni_tag + cfg.UNIQUE_TAG].astype(int, errors='ignore')
                mapping = mapping.set_index(tag)[cfg.TAG_REGIONE]
                self.not_match[self.regioni_tag + cfg.UNIQUE_TAG] = self.not_match[
                    self.regioni_tag + cfg.UNIQUE_TAG].map(mapping)
            self.not_match[self.regioni_tag + cfg.UNIQUE_TAG] = _clean_denom_text(
                self.not_match[self.regioni_tag + cfg.UNIQUE_TAG])

    @validate
    def get_not_matched_list(self) -> List[str]:
        if self.not_match is None:
            raise Exception("Run simple match before getting list of not matched values.")
        result = list(self.not_match[self.not_match[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique())
        return result

    @validate
    def get_n_not_matched(self) -> int:
        if self.not_match is None:
            raise Exception("Run simple match before getting number of not matched values.")
        result = self.not_match[self.not_match[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].nunique()
        return result

    def _test_if_df_contains_homonym_comuni(self):
        info_tag_details = cfg.TAG_SIGLA
        if self.province_code is not None:
            info_tag_details = _get_tag_anag(self.province_code, cfg.LEVEL_PROVINCIA)
            self.df = self._split_denom_comuni_omonimi(self.df, cfg.KEY_UNIQUE, self.province_tag, info_tag_details)
        elif self.regioni_code is not None:
            info_tag_details = _get_tag_anag(self.regioni_code, cfg.LEVEL_REGIONE)
            self.df = self._split_denom_comuni_omonimi(self.df, cfg.KEY_UNIQUE, self.regioni_tag, info_tag_details)
        else:
            if (self.df[cfg.KEY_UNIQUE].isin([a.lower() for a in cfg.comuni_omonimi[cfg.TAG_COMUNE]])).sum() > 0:
                log.warning(
                    "The dataset contains some homonym comune. You can distinguish them only by using another gerachical information (ex.: provincia or regione).")
        self.info_df = self._split_denom_comuni_omonimi(self.info_df, cfg.KEY_UNIQUE, info_tag_details,
                                                        info_tag_details, log_flag=False)

    @staticmethod
    def _split_denom_comuni_omonimi(df, den_tag, details_tag1, details_tag2, log_flag=True):
        split_df = pd.DataFrame.from_dict(cfg.comuni_omonimi)
        split_df[cfg.TAG_COMUNE] = split_df[cfg.TAG_COMUNE].str.lower()
        split_df[details_tag2] = split_df[details_tag2].astype(str).str.lower()
        split_df["key"] = split_df[cfg.TAG_COMUNE] + split_df[details_tag2]
        pos = df[den_tag].str.lower().isin(split_df[cfg.TAG_COMUNE].unique())
        if (pos.sum() > 0) & log_flag:
            homonimus_coumni = df[pos][den_tag].unique()
            log.info(
                "Found {} homonimus comuni in dataset: {}. Those will correctly matched by {}".format(
                    len(homonimus_coumni),
                    homonimus_coumni,
                    details_tag1))
        df[den_tag] = np.where(pos, df[den_tag].str.lower() + df[details_tag1].astype(str).str.lower(),
                               df[den_tag])
        df[den_tag] = df[den_tag].replace(split_df.set_index("key")["new_name"])
        return df

    def _find_any_bilingual_name(self):
        if self.level == cfg.LEVEL_COMUNE:
            replace_multilanguage_name = get_double_languages_mapping_comuni(self.info_df)

        elif self.level == cfg.LEVEL_PROVINCIA:
            replace_multilanguage_name = get_double_languages_mapping_province(self.info_df)
        elif self.level == cfg.LEVEL_REGIONE:
            replace_multilanguage_name = get_double_languages_mapping_regioni(self.info_df)
        else:
            return
        for k, v in replace_multilanguage_name.items():
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.replace(f"\b{k}\b",
                                                                          v,
                                                                          regex=True)
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.replace(r"(?s)([A-Za-z].*) ?[\/] ?\1",
                                                                      r"\1",
                                                                      regex=True)

    def _rename_any_english_name(self):
        s = self.df[cfg.KEY_UNIQUE].replace(cfg.rename_english_name)
        different_pos = (s != self.df[cfg.KEY_UNIQUE]) & self.df[cfg.KEY_UNIQUE].notnull()
        if different_pos.any():
            replaces = self.df[different_pos][cfg.KEY_UNIQUE].unique()
            log.info("Replaced {} name written in english:{}".format(len(replaces), replaces))
            self.df[cfg.KEY_UNIQUE] = s

    def _clean_province_name(self):
        for k, v in cfg.clear_provnce_name.items():
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.replace(k, v)

    def _clean_comuni_name(self):
        for k, v in cfg.clear_comuni_name.items():
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.replace(k, v)

    def _find_any_variation_from_istat_history(self):
        df_variazioni = get_administrative_changes_df()
        df_variazioni[cfg.TAG_COMUNE] = _clean_denom_text(df_variazioni[cfg.TAG_COMUNE])
        df_variazioni["new_denominazione_comune"] = _clean_denom_text(df_variazioni["new_denominazione_comune"])
        df_variazioni["data_decorrenza"] = pd.to_datetime(df_variazioni["data_decorrenza"])
        df_variazioni.sort_values([cfg.TAG_COMUNE, "data_decorrenza"], ascending=False, inplace=True)
        df_variazioni = df_variazioni.groupby(cfg.TAG_COMUNE)["new_denominazione_comune"].last().to_dict()
        s = np.where(self.df[cfg.KEY_UNIQUE].isin(list(self.info_df[cfg.KEY_UNIQUE].unique())),
                     self.df[cfg.KEY_UNIQUE],
                     self.df[cfg.KEY_UNIQUE].replace(df_variazioni))
        if (s != self.df[cfg.KEY_UNIQUE]).any():
            replaces = self.df[(s != self.df[cfg.KEY_UNIQUE])][cfg.KEY_UNIQUE].unique()
            log.info("Match {} name that are no longer comuni:\n{}".format(len(replaces), replaces))
            self.df[cfg.KEY_UNIQUE] = s

    def _try_custom_replace_denom(self):
        list_den_anagrafica = self.list_anag
        list_den_not_found = self.get_not_matched_list()
        dict_den_anag = {self._custom_replace_denom(a): a for a in list_den_anagrafica}
        dict_den_not_found = {a: self._custom_replace_denom(a) for a in list_den_not_found}
        dict_den_not_found = {k: dict_den_anag[v] for k, v in dict_den_not_found.items() if v in dict_den_anag.keys()}

        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace(dict_den_not_found)
        _ = self._check_non_match()

    @staticmethod
    def _custom_replace_denom(value):
        for v in cfg.clear_den_replace:
            value = value.replace(v[0], v[1])
        value = " ".join(value.split())
        return value

    @staticmethod
    def _get_info_from_address(address, regioni, province, comuni):
        regione, provincia, comune = None, None, None
        extract = re.search(
            '(?P<comune2>[^,]+, )?(?P<comune1>[^,]+), (?P<provincia>[^,]+), (?P<regione>[^,0-9]+)(?P<cap>, [0-9]{5})?, Italia',
            address)
        if extract:
            regione = _clean_denom_text_value(extract.group("regione"))
            if regione not in regioni:
                regione = None
            provincia = extract.group("provincia")
            provincia = _clean_denom_text_value(provincia.replace("Roma Capitale", "Roma"))
            if provincia not in province:
                provincia = None
            comune = _clean_denom_text_value(extract.group("comune1"))
            if (comune is None) or (comune not in comuni):
                comune = None
                comune2 = extract.group("comune2")
                if comune2:
                    comune = _clean_denom_text_value(comune2[:-2])
                    if (comune is None) or (comune not in comuni):
                        comune = None
                if (comune is None) and (provincia in comuni):
                    comune = provincia
        if (regione is not None) and (provincia is not None) and (comune is not None):
            return regione, provincia, comune
        else:
            return None, None, None

    def run_find_frazioni(self):
        if (self.code != cfg.CODE_DENOMINAZIONE) & (self.level != cfg.LEVEL_COMUNE):
            raise Exception("Run find_frazioni only to find comune by denomination.")
        if self.not_match is None:
            raise Exception("Run simple match before.")
        geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)  # "trial"
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        regioni = [_clean_denom_text_value(a) for a in self.regioni]
        province = [_clean_denom_text_value(a) for a in self.province]
        comuni = [_clean_denom_text_value(a) for a in self.comuni]
        match_dict = {}
        n = self.get_n_not_matched()
        log.info(f"Needed at least {n} seconds")
        if (n > 0) & ((self.province_tag is not None) | (self.regioni_tag is not None)):
            self._clean_hierarchical_structure()
        for index, row in self.not_match.iterrows():
            check_flag = False
            check = None
            if (self.province_tag is not None) and (row[self.province_tag] is not None):
                check_flag = True
                check = row[self.province_tag + cfg.UNIQUE_TAG]
                check_level = cfg.LEVEL_PROVINCIA
                mapping_value = row[self.province_tag]
            elif (self.regioni_tag is not None) and (row[self.regioni_tag] is not None):
                check_flag = True
                check = row[self.regioni_tag + cfg.UNIQUE_TAG]
                mapping_value = row[self.regioni_tag]
            el = row[cfg.KEY_UNIQUE]
            if check_flag:
                p = geocode(f"{el}, {check}, italia")
            else:
                p = geocode(f"{el}, italia")
            if p is not None:
                address = p.address
                if el in address.lower():
                    regione, provincia, comune = self._get_info_from_address(address, regioni, province, comuni)
                    if comune is not None:
                        if not check_flag:
                            self.df[cfg.KEY_UNIQUE].replace(el, comune, inplace=True)
                            match_dict[el] = comune
                        elif check_level == cfg.LEVEL_PROVINCIA:
                            if provincia == check:
                                match_dict[f"{el}+{provincia}"] = comune
                                pos = (self.df[cfg.KEY_UNIQUE] == el) & (
                                        self.df[self.province_tag] == mapping_value)
                                self.df.loc[pos, cfg.KEY_UNIQUE] = comune
                        elif check_level == cfg.LEVEL_REGIONE:
                            if regione == check:
                                match_dict["{}+{}".format(el, regione)] = comune
                                pos = (self.df[cfg.KEY_UNIQUE] == el) & (self.df[self.regioni_tag] == mapping_value)
                                self.df.loc[pos, cfg.KEY_UNIQUE] = comune

        if len(match_dict) > 0:
            log.info("Match {} name that corrisponds to a possible frazione of a comune:\n{}".format(len(match_dict),
                                                                                                     {k.split("-")[0]: v
                                                                                                      for k, v in
                                                                                                      match_dict.items()}))
            _ = self._check_non_match()

    def _get_cleaned_geo_info(self, comune, level):
        mapping = self.info_df.set_index(cfg.TAG_COMUNE, _get_tag_anag(cfg.CODE_DENOMINAZIONE, level)).to_dict()
        geo_info = mapping[comune]
        return _clean_denom_text_value(geo_info)

    @validate
    def run_find_frazioni_from_google(self, n_url_read: int = 1):
        comuni = [_clean_denom_text_value(a) for a in self.comuni]
        match_dict = {}
        n = self.get_n_not_matched()
        log.info(f"Needed at least {n} seconds")
        if (n > 0) & ((self.province_tag is not None) | (self.regioni_tag is not None)):
            self._clean_hierarchical_structure()
        for index, row in self.not_match.iterrows():
            check_flag = False
            check = None
            if (self.province_tag is not None) and (row[self.province_tag] is not None):
                check_flag = True
                check = row[self.province_tag + cfg.UNIQUE_TAG]
                check_level = cfg.LEVEL_PROVINCIA
                mapping_value = row[self.province_tag]
            elif (self.regioni_tag is not None) and (row[self.regioni_tag] is not None):
                check_flag = True
                check = row[self.regioni_tag + cfg.UNIQUE_TAG]
                mapping_value = row[self.regioni_tag]
            el = row[cfg.KEY_UNIQUE]
            if check_flag:
                query = """ "{}" è una frazione del comune di {}""".format(el, check)
            else:
                query = """ "{}" è una frazione del comune di""".format(el)
            match_comuni = []
            urls = []
            try:
                urls = list(
                    search(query, tld='com', num=n_url_read, lang="it", country="Italy", stop=n_url_read, pause=2.5,
                           verify_ssl=False))
            except Exception as e:
                log.error('Failed to search on google tentative 1: ' + str(e))
                try:
                    my_results = google_query(query,
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
                    output = ''
                    blacklist = [
                        '[document]',
                        'noscript',
                        'header',
                        'html',
                        'meta',
                        'head',
                        'input',
                        'script',
                        # there may be more elements you don't want, such as "style", etc.
                    ]

                    for t in text:
                        if t.parent.name not in blacklist:
                            output += '{} '.format(t)
                    output = _clean_htmltext(output)
                    r1 = re.findall(cfg.regex_find_frazioni.format(el), output)
                    for r in r1:
                        find_comuni = r[8].split(" ")
                        test_name = [" ".join(find_comuni[:i + 1]) for i in range(len(find_comuni))]
                        test_list = [a in comuni for a in test_name]
                        if any(test_list):
                            match_comuni.extend([test_name[i] for i in range(len(find_comuni)) if test_list[i]])
            match_comuni = list(set(match_comuni))
            if not check_flag:
                if len(match_comuni) == 1:
                    match_dict[el] = match_comuni[0]
                    self.df[cfg.KEY_UNIQUE].replace(el, match_comuni[0], inplace=True)
            else:
                if check_level == cfg.LEVEL_PROVINCIA:
                    test_list = [a for a in match_comuni if
                                 self._get_cleaned_geo_info(a, cfg.LEVEL_PROVINCIA) == check]
                    if len(test_list) == 1:
                        match_dict["{}+{}".format(el, check)] = test_list[0]
                        pos = (self.df[cfg.KEY_UNIQUE] == el) & (self.df[self.province_tag] == mapping_value)
                        self.df.loc[pos, cfg.KEY_UNIQUE] = test_list[0]
                elif check_level == cfg.LEVEL_REGIONE:
                    test_list = [a for a in match_comuni if
                                 self._get_cleaned_geo_info(a, cfg.LEVEL_REGIONE) == check]
                    if len(test_list) == 1:
                        match_dict["{}+{}".format(el, check)] = test_list[0]
                        pos = (self.df[cfg.KEY_UNIQUE] == el) & (self.df[self.regioni_tag] == check)
                        self.df.loc[pos, cfg.KEY_UNIQUE] = test_list[0]
        if len(match_dict) > 0:
            log.info("Match {} name that corrisponds to a possible frazione of a comune from google:\n{}".format(
                len(match_dict), match_dict))
            _ = self._check_non_match()

    @validate
    def get_result(self, add_missing: bool = False, drop_not_match: bool = False) -> pd.DataFrame:
        if self.not_match is None:
            raise Exception("Run simple match before get the result.")
        n_not_match = self.not_match[cfg.KEY_UNIQUE].nunique()
        if n_not_match > 0:
            log.warning("Unable to find {} {}: {}".format(n_not_match, self.geo_tag_anag, self.get_not_matched_list()))
        else:
            log.info("Found every {}".format(self.geo_tag_anag))

        if self.geo_tag_anag in self.df.columns:
            self.df.rename(columns={self.geo_tag_anag: self.geo_tag_anag + "_original"}, inplace=True)

        list_col = list(set(self.keys + [cfg.TAG_COMUNE, cfg.TAG_CODICE_COMUNE,
                                         cfg.TAG_PROVINCIA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
                                         cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE,
                                         cfg.TAG_AREA_GEOGRAFICA,
                                         cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]))
        if add_missing:
            if drop_not_match:
                how = "left"
            else:
                how = "outer"
            result = self.info_df.merge(self.df, on=cfg.KEY_UNIQUE, how=how)
            list_col = [col for col in list_col if col in result.columns]
            result = result[list_col].merge(self.original_df, on=self.keys, how=how, suffixes=["_new", ""])
        else:
            if drop_not_match:
                how = "inner"
            else:
                how = "left"
            result = self.df.merge(self.info_df, on=cfg.KEY_UNIQUE, how=how)
            list_col = [col for col in list_col if col in result.columns]
            index_name = self.original_df.index.name
            if index_name is None:
                index_name = "index"
            result = self.original_df.reset_index().merge(result[list_col], on=self.keys, how=how,
                                                          suffixes=["", "_new"]).set_index(index_name)

        return result

    @validate
    def run_similarity_match(self, unique_flag: bool = False):
        if unique_flag:
            input_den = self.df[cfg.KEY_UNIQUE].values()
            not_match2 = [a for a in self.list_anag if a not in input_den]
            match_dict = self._find_match(not_match2, self.get_not_matched_list(), unique=True)
            self.similarity_dict = {v[0]: (k, v[1]) for k, v in match_dict.items()}
        else:
            self.similarity_dict = self._find_match(self.get_not_matched_list(), self.list_anag)
        n = len(self.similarity_dict)
        if n > 1:
            log.info("Match {} name by similarity:\n{}".format(n, self.similarity_dict))
        else:
            log.info("No match by similarity")

    def get_similairty_result(self):
        return self.similarity_dict

    def accept_similarity_result(self):
        if self.similarity_dict is not None:
            _ = self._check_non_match()
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace(
                {k: v[0] for k, v in self.similarity_dict.items()})
        else:
            raise Exception("Run  run_similarity_match before accept_similarity_result.")

    @validate
    def use_manual_match(self, manual_dict: Dict[str, str]):
        _ = self._check_non_match()
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace(manual_dict)

    @staticmethod
    def _find_match(not_match1, not_match2, unique=False) -> Dict[str, str]:
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
                if score > cfg.min_acceptable_similarity:
                    match_dict[a] = (best_match, score)
                    if unique:
                        not_match2.remove(best_match)
        return match_dict


def _get_tag_anag(code, level):
    if level == cfg.LEVEL_COMUNE:
        if code == cfg.CODE_CODICE_ISTAT:
            result = cfg.TAG_CODICE_COMUNE
        else:
            result = cfg.TAG_COMUNE
    elif level == cfg.LEVEL_PROVINCIA:
        if code == cfg.CODE_CODICE_ISTAT:
            result = cfg.TAG_CODICE_PROVINCIA
        elif code == cfg.CODE_SIGLA:
            result = cfg.TAG_SIGLA
        else:
            result = cfg.TAG_PROVINCIA
    elif level == cfg.LEVEL_REGIONE:
        if code == cfg.CODE_CODICE_ISTAT:
            result = cfg.TAG_CODICE_REGIONE
        else:
            result = cfg.TAG_REGIONE
    else:
        raise Exception("Level UNKNOWN")
    return result


def _code_or_desc(list_values):
    list_values = [x for x in list_values if (str(x) != 'nan') and (x is not None)]
    n_tot = len(list_values)
    if n_tot == 0:
        result = cfg.CODE_DENOMINAZIONE
    elif (sum([isinstance(item, int) or isinstance(item, float) or item.isdigit() for item in
               list_values]) / n_tot) > 0.8:
        result = cfg.CODE_CODICE_ISTAT
    elif (sum([isinstance(item, str) and item.isalpha() and len(item) == 2 for item in list_values]) / n_tot) > 0.8:
        result = cfg.CODE_SIGLA
    else:
        result = cfg.CODE_DENOMINAZIONE
    return result


def _clean_denom_text(series):
    series = series.fillna("")
    series = series.astype(str)
    series = series.where(series != "", None)
    series = series.str.lower()  # All strig in lowercase
    series = series.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non alphabetic characters
    series = series.str.strip()
    series = series.str.replace(r'\s+', ' ', regex=True)
    series = series.replace(cfg.comuni_exceptions)
    series = series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')  # Remove accent
    series = series.replace(cfg.comuni_exceptions)
    return series


def _clean_denom_text_value(value):
    value = value.lower()  # All strig in lowercase
    value = re.sub(r'[^\w\s]', ' ', value)  # Remove non alphabetic characters
    value = value.strip()
    value = re.sub(r'\s+', ' ', value)
    cfg.comuni_exceptions.get(value, value)
    value = unidecode.unidecode(value)
    cfg.comuni_exceptions.get(value, value)
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
def get_geo_info_from_comune(comune: str, provincia: str = None, regione: str = None) -> Dict[str, str]:
    df = pd.DataFrame(data=[[comune, provincia, regione]], columns=["comune", "provincia", "regione"])
    addInfo = AddGeographicalInfo(df)
    addInfo.set_comuni_tag("comune")
    if provincia:
        addInfo.set_province_tag("provincia")
    elif regione:
        addInfo.set_regioni_tag("regione")
    addInfo.run_simple_match()
    addInfo.run_find_frazioni()
    df = addInfo.get_result()
    if df[cfg.TAG_COMUNE].values[0] is None:
        raise Exception("Unable to find the city {}".format(comune))
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
        df["address_search"] = np.where(t, df["address_search"], df["address_search"] + ", " + df[comuni_tag].str.lower())

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
        self.regioni_code = _code_or_desc(list(self.original_df[col_name].unique()))
        self.regioni_result_tag = _get_tag_anag(self.regioni_code, cfg.LEVEL_REGIONE)

    @validate
    def set_comuni_tag(self, col_name: str, use_for_check_nation: bool = False):
        self.use_for_check_nation = use_for_check_nation
        _test_column_in_dataframe(self.original_df, col_name)
        self.comuni_tag = col_name
        self.comuni_code = _code_or_desc(list(self.original_df[col_name].unique()))
        self.comuni_result_tag = _get_tag_anag(self.comuni_code, cfg.LEVEL_COMUNE)

    @validate
    def set_province_tag(self, col_name: str):
        _test_column_in_dataframe(self.original_df, col_name)
        self.province_tag = col_name
        self.province_code = _code_or_desc(list(self.original_df[col_name].unique()))
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
