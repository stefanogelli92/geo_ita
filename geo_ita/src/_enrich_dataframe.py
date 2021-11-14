import difflib
import math
import os
import time
import logging
import ssl
from pathlib import PureWindowsPath
from datetime import datetime

import unidecode
from geo_ita.src.definition import *

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
import geopandas as gpd
from geopy.distance import distance
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
# from sklearn.neighbors import KernelDensity

from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re

from geo_ita.src._data import (get_df, get_df_comuni, get_variazioni_amministrative_df, _get_list,
    get_double_languages_mapping, _get_shape_italia, get_high_resolution_population_density_df)
import geo_ita.src.config as cfg

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx


from googleapiclient.discovery import build
api_key = "AIzaSyDQTlp0F0tExsXg1PZn1FF_ugK_bwBhlqA"
cse_id = "8558e6507a2bf9d77"


def google_query(query, api_key, cse_id, **kwargs):
    query_service = build("customsearch",
                          "v1",
                          developerKey=api_key
                          )
    query_results = query_service.cse().list(q=query,    # Query
                                             cx=cse_id,  # CSE ID
                                             **kwargs
                                             ).execute()
    return query_results['items']


def _clean_htmltext(text):
    text = text.lower()
    text = re.sub('\s+', ' ', text)
    text = re.sub('[^A-Za-z0-9.]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text


def _test_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise Exception("Pass a Pandas DataFrame as parameter.")

def _test_column_in_dataframe(df, col):
    if col not in df.columns:
        raise Exception("Column {} not found in DataFrame.".format(col))

class AddGeographicalInfo:

    def __init__(self, df):
        self.original_df = df
        _test_dataframe(self.original_df)
        self.keys = None
        self.df = None
        self.comuni, self.province, self.sigle, self.regioni = _get_list()
        self._find_info()
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

    def _find_info(self):
        pass
        #for col in self.df.select_dtypes(include='object').columns:

    def set_comuni_tag(self, col_name):
        _test_column_in_dataframe(self.original_df, col_name)
        self.comuni_tag = col_name
        self.comuni_code = _code_or_desc(list(self.original_df[col_name].unique()))
        self.level = cfg.LEVEL_COMUNE
        self.geo_tag_input = col_name
        self.code = self.comuni_code

    def set_province_tag(self, col_name):
        _test_column_in_dataframe(self.original_df, col_name)
        self.province_tag = col_name
        self.province_code = _code_or_desc(list(self.original_df[col_name].unique()))
        if (self.level != cfg.LEVEL_COMUNE) & (self.level != cfg.LEVEL_CAP):
            self.level = cfg.LEVEL_PROVINCIA
            self.geo_tag_input = col_name
            self.code = self.province_code

    def set_regioni_tag(self, col_name):
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

    def _run_sigla(self):
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.lower()
        self.info_df[cfg.KEY_UNIQUE] = self.info_df[cfg.KEY_UNIQUE].str.lower()
        list_input = self.df[self.df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique()
        self.list_anag = self.info_df[self.info_df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique()
        n_tot = len(list_input)
        self.not_match = list(set(list_input) - set(self.list_anag))
        n_not_match = len(self.not_match)

        if n_not_match == 0:
            log.info("Matching completed, found {} different sigle.".format(n_tot))
        else:
            log.warning("Matched {} over {}.".format(n_tot - n_not_match, n_tot))

    def _run_codice(self):
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].astype(int, errors='ignore')
        self.info_df[cfg.KEY_UNIQUE] = self.info_df[cfg.KEY_UNIQUE].astype(int, errors='ignore')
        list_input = self.df[self.df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique()
        self.list_anag = self.info_df[self.info_df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique()
        n_tot = len(list_input)
        self.not_match = list(set(list_input) - set(self.list_anag))
        n_not_match = len(self.not_match)

        if n_not_match == 0:
            log.info("Matching completed, found {} different codes.".format(n_tot))
        else:
            log.warning("Matched {} over {}.".format(n_tot - n_not_match, n_tot))

    def _run_denominazione(self):
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.lower()
        if self.level == cfg.LEVEL_COMUNE:
            self._test_if_df_contains_homonym_comuni()

        self._find_any_bilingual_name()

        self.df[cfg.KEY_UNIQUE] = _clean_denom_text(self.df[cfg.KEY_UNIQUE])
        self.info_df[cfg.KEY_UNIQUE] = _clean_denom_text(self.info_df[cfg.KEY_UNIQUE])

        if self.level == cfg.LEVEL_COMUNE:
            self._rename_any_english_name()
            self._find_any_variation_from_istat_history()

        list_den_input = self.df[self.df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique()
        self.list_anag = self.info_df[self.info_df[cfg.KEY_UNIQUE].notnull()][cfg.KEY_UNIQUE].unique()

        n_tot = len(list_den_input)

        self.not_match = list(set(list_den_input) - set(self.list_anag))

        self._try_custom_replace_denom()

        n_not_match = len(self.not_match)

        if n_not_match == 0:
            log.info("Matching completed, found {} different names.".format(n_tot))
        else:
            log.warning("Matched {} over {}.".format(n_tot - n_not_match, n_tot))

    def get_not_matched(self):
        if self.not_match is None:
            raise Exception("Run simple match before get list of not matched values.")
        return self.not_match

    def _test_if_df_contains_homonym_comuni(self):
        info_tag_details = cfg.TAG_SIGLA
        if self.province_code is not None:
            info_tag_details = _get_tag_anag(self.province_code, cfg.LEVEL_PROVINCIA)
            self.df = self._split_denom_comuni_omonimi(self.df, cfg.KEY_UNIQUE, self.province_tag, info_tag_details)
        elif self.regioni_code is not None:
            info_tag_details = _get_tag_anag(self.regioni_code, cfg.LEVEL_REGIONE)
            self.df = self._split_denom_comuni_omonimi(self.df, cfg.KEY_UNIQUE, self.regioni_tag, info_tag_details)
        else:
            # TODO Loggare warning che se ci sono comuni omonimi non sarà in grado di abbinarli correttamente
            pass
        self.info_df = self._split_denom_comuni_omonimi(self.info_df, cfg.KEY_UNIQUE, info_tag_details, info_tag_details, log_flag=False)

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
        if self.comuni_code == cfg.CODE_DENOMINAZIONE:
            replace_multilanguage_name = get_double_languages_mapping()
            for k, v in replace_multilanguage_name.items():
                self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.replace(r"\b{}\b".format(k),
                                                                              v,
                                                                              regex=True)
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].str.replace(r"([A-Za-z](?s).*) ?[-\/] ?\1",
                                                                          r"\1",
                                                                          regex=True)
        else:
            # TODO Gestire pulizia pure per province e regioni
            pass

    def _rename_any_english_name(self):
        s = self.df[cfg.KEY_UNIQUE].replace(cfg.rename_comuni_nomi)
        if (s != self.df[cfg.KEY_UNIQUE]).any():
            replaces = self.df[(s != self.df[cfg.KEY_UNIQUE])][cfg.KEY_UNIQUE].unique()
            log.info("Replaced {} comuni written in english:{}".format(len(replaces), replaces))
            self.df[cfg.KEY_UNIQUE] = s

    def _find_any_variation_from_istat_history(self):
        df_variazioni = get_variazioni_amministrative_df()
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
        list_den_not_found = self.not_match
        dict_den_anag = {self._custom_replace_denom(a): a for a in list_den_anagrafica}
        dict_den_not_found = {a: self._custom_replace_denom(a) for a in list_den_not_found}
        dict_den_not_found = {k: dict_den_anag[v] for k, v in dict_den_not_found.items() if v in dict_den_anag.keys()}

        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace(dict_den_not_found)

        self.not_match = [a for a in self.not_match if a not in dict_den_not_found.keys()]

    @staticmethod
    def _custom_replace_denom(value):
        for v in cfg.clear_den_replace:
            value = value.replace(v[0], v[1])
        value = " ".join(value.split())
        return value

    def run_find_frazioni(self):
        geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)  # "trial"
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        regioni = [_clean_denom_text_value(a) for a in self.regioni]
        province = [_clean_denom_text_value(a) for a in self.province]
        comuni = [_clean_denom_text_value(a) for a in self.comuni]
        match_dict = {}
        n = len(self.not_match)
        log.info("Needed at least {} seconds".format(n))
        for el in self.not_match:
            p = geocode(el + ", italia")
            if p is not None:
                address = p.address
                if el in address.lower():
                    extract = re.search(
                        '(?P<comune2>[^,]+, )?(?P<comune1>[^,]+), (?P<provincia>[^,]+), (?P<regione>[^,0-9]+)(?P<cap>, [0-9]{5})?, Italia',
                        address)
                    if extract:  # and not any(word.lower() in address.lower() for word in ["via", "viale", "piazza"]):
                        regione = _clean_denom_text_value(extract.group("regione"))
                        provincia = extract.group("provincia")
                        provincia = _clean_denom_text_value(provincia.replace("Roma Capitale", "Roma"))
                        comune = _clean_denom_text_value(extract.group("comune1"))
                        comune2 = extract.group("comune2")
                        if comune2:
                            comune2 = _clean_denom_text_value(comune2[:-2])
                        if (regione in regioni) & (provincia in province):
                            if comune in comuni:
                                match_dict[el] = comune
                            elif (comune2 is not None) and (comune2 in comuni):
                                match_dict[el] = comune2
                            elif provincia in comuni:
                                match_dict[el] = provincia
        if len(match_dict) > 0:
            log.info("Match {} name that corrisponds to a possible frazione of a comune:\n{}".format(len(match_dict), match_dict))
            self.not_match = [x for x in self.not_match if x not in match_dict.keys()]
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace(match_dict)
        if self.frazioni_dict is not None:
            self.frazioni_dict.update(match_dict)
        else:
            self.frazioni_dict = match_dict


    def run_find_frazioni_from_google(self, n_url_read=1):
        comuni = [_clean_denom_text_value(a) for a in self.comuni]
        match_dict = {}
        for el in self.not_match:
            query = """ "{}" è una frazione del comune di""".format(el)
            match_comuni = []
            urls = []
            try:
                urls = list(search(query, tld='com', num=n_url_read, lang="it", country="Italy", stop=n_url_read, pause=2.5, verify_ssl=False))
            except Exception as e:
                log.error('Failed to search on google tentative 1: ' + str(e))
                try:
                    my_results = google_query(query,
                                              api_key,
                                              cse_id,
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
            if len(match_comuni) == 1:
                match_dict[el] = match_comuni[0]
        if len(match_dict) > 0:
            log.info("Match {} name that corrisponds to a possible frazione of a comune from google:\n{}".format(len(match_dict), match_dict))
            self.not_match = [x for x in self.not_match if x not in match_dict.keys()]
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace(match_dict)
        if self.frazioni_dict is not None:
            self.frazioni_dict.update(match_dict)
        else:
            self.frazioni_dict = match_dict

    def get_result_frazioni(self):
        return self.frazioni_dict

    def get_result(self, add_missing=False, drop_not_match=False):
        if self.not_match is None:
            raise Exception("Run simple match before get the result.")
        if len(self.not_match) > 0:
            log.warning("Unable to find {} {}: {}".format(len(self.not_match), self.geo_tag_anag, self.not_match))
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
            result = self.original_df.merge(result[list_col], on=self.keys, how=how, suffixes=["", "_new"])

        return result

    def run_similarity_match(self, unique_flag=False):
        if unique_flag:
            input_den = self.df[cfg.KEY_UNIQUE].values()
            not_match2 = [a for a in self.list_anag if a not in input_den]
            match_dict = self._find_match(not_match2, self.not_match, unique=True)
            self.similarity_dict = {v[0]: (k, v[1]) for k, v in match_dict.items()}
        else:
            self.similarity_dict = self._find_match(self.not_match, self.list_anag)
        n = len(self.similarity_dict)
        if n > 1:
            log.info("Match {} name by similarity:\n{}".format(n, self.similarity_dict))
        else:
            log.info("No match by similarity")

    def get_similairty_result(self):
        return self.similarity_dict

    def accept_similarity_result(self):
        if self.similarity_dict is not None:
            self.not_match = [x for x in self.not_match if x not in self.similarity_dict.keys()]
            self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace({k: v[0] for k, v in self.similarity_dict.items()})
        else:
            raise Exception("Run  run_similarity_match before accept_similarity_result.")

    def use_manual_match(self, manual_dict):
        if not isinstance(manual_dict, dict):
           raise Exception("Pass a dictionary (name -> replace).")
        self.not_match = [x for x in self.not_match if x not in manual_dict.keys()]
        self.df[cfg.KEY_UNIQUE] = self.df[cfg.KEY_UNIQUE].replace(manual_dict)

    @staticmethod
    def _find_match(not_match1, not_match2, unique=False):
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
    elif (sum([isinstance(item, np.integer) or isinstance(item, np.float) or item.isdigit() for item in list_values]) / n_tot) > 0.8:
        result = cfg.CODE_CODICE_ISTAT
    elif (sum([isinstance(item, str) and item.isalpha() and len(item) == 2 for item in list_values]) / n_tot) > 0.8:
        result = cfg.CODE_SIGLA
    else:
        result = cfg.CODE_DENOMINAZIONE
    return result


def _clean_denom_text(series):
    series = series.str.lower()  # All strig in lowercase
    series = series.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non alphabetic characters
    series = series.str.strip()
    series = series.str.replace(r'\s+', ' ', regex=True)
    series = series.replace(cfg.comuni_exceptions)
    series = series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')  # Remove accent
    series = series.replace(cfg.comuni_exceptions)
    #for v in cfg.clear_den_replace:
    #    series = series.str.replace(v[0], v[1])
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
            #df.loc[(df[long_tag].isna()) | (df[lat_tag].isna()), "geometry"] = None
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


def get_geo_info_from_comune(comune, provincia=None, regione=None):
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
    result_dict = {
        "comune": df[cfg.TAG_COMUNE].values[0],
        "provincia": df[cfg.TAG_PROVINCIA].values[0],
        "sigla": df[cfg.TAG_SIGLA].values[0],
        "regione": df[cfg.TAG_REGIONE].values[0],
        "area_geografica": df[cfg.TAG_AREA_GEOGRAFICA].values[0],
        "popolazione": df[cfg.TAG_POPOLAZIONE].values[0],
        "superficie": df[cfg.TAG_SUPERFICIE].values[0]
    }
    return result_dict


def get_geo_info_from_regione(regione):
    df = pd.DataFrame(data=[[regione]], columns=["regione"])
    addInfo = AddGeographicalInfo(df)
    addInfo.set_regioni_tag("regione")
    addInfo.run_simple_match()
    addInfo.run_find_frazioni()
    df = addInfo.get_result()
    if pd.isna(df[cfg.TAG_REGIONE].values[0]):
        raise Exception("Unable to find the region {}".format(regione))
    result_dict = {
        "regione": df[cfg.TAG_REGIONE].values[0],
        "area_geografica": df[cfg.TAG_AREA_GEOGRAFICA].values[0],
        "popolazione": df[cfg.TAG_POPOLAZIONE].values[0],
        "superficie": df[cfg.TAG_SUPERFICIE].values[0]
    }
    return result_dict


def get_geo_info_from_provincia(provincia, regione=None):
    df = pd.DataFrame(data=[[provincia, regione]], columns=["provincia", "regione"])
    addInfo = AddGeographicalInfo(df)
    addInfo.set_province_tag("provincia")
    if regione:
        addInfo.set_regioni_tag("regione")
    addInfo.run_simple_match()
    addInfo.run_find_frazioni()
    df = addInfo.get_result()
    if df[cfg.TAG_PROVINCIA].values[0] is None:
        raise Exception("Unable to find the city {}".format(provincia))
    result_dict = {
        "provincia": df[cfg.TAG_PROVINCIA].values[0],
        "sigla": df[cfg.TAG_SIGLA].values[0],
        "regione": df[cfg.TAG_REGIONE].values[0],
        "area_geografica": df[cfg.TAG_AREA_GEOGRAFICA].values[0],
        "popolazione": df[cfg.TAG_POPOLAZIONE].values[0],
        "superficie": df[cfg.TAG_SUPERFICIE].values[0]
    }
    return result_dict


def get_city_from_coordinates(df0, latitude_columns=None, longitude_columns=None):
    _test_dataframe(df0)
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
    df = df.to_crs({'init': 'epsg:4326'})

    n_tot = df.shape[0]
    map_city = gpd.sjoin(df, df_comuni, op='within', how="left")

    missing = list(map_city[map_city[cfg.TAG_COMUNE].isna()]["geometry"].unique())
    if len(missing) == 0:
        log.info("Found the correct city for each point")
    else:
        log.warning("Unable to find the city for {} points: {}".format(len(missing), [(x.x, x.y) for x in missing]))
    map_city = map_city[["key_mapping", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]]
    return df0.merge(map_city, on=["key_mapping"], how="left").drop(["key_mapping"], axis=1)


def __test_city_in_address(df, city_tag, address_tag):
    return df.apply(lambda x: x[city_tag].lower() in x[address_tag] if (x[address_tag] and x[city_tag]) else False, axis=1)


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
                                              api_key,
                                              cse_id,
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
                    pattern = "{prefix} ?{abbreviations} ?{suffix}".format(prefix=prefix,
                                                                           abbreviations=abbreviations,
                                                                           suffix=suffix)
                    r1 = re.search(pattern, output)
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


def _try_wrong_replace_of_accents(df, address_tag, geocode):
    log.info("Try to find _wrong_replace_of_accents")
    regex =r"\b(del|dell|d|nell|sull|sant|Sant)([A-Z][^\s]+)"
    pos_replace = (df["location"].isna() &
                   df[address_tag].str.contains(regex))
    not_found = list(df.loc[pos_replace, address_tag].unique())
    for address in not_found:
        new_name = re.sub(regex, r"\1'\2", address)
        pos_match = df[address_tag] == address
        df.loc[pos_match, "address_search"] = new_name
        location = geocode(new_name)
        if location is not None:
            df.loc[pos_match, "location"] = location
            df.loc[pos_match, "latitude"] = location.latitude
            df.loc[pos_match, "longitude"] = location.longitude
            df.loc[pos_match, "address_test"] = location.address.lower()
    return df


def get_coordinates_from_address(df0, address_tag, city_tag=None, province_tag=None, regione_tag=None, n_url_read=1):
    # TODO add successive tentative (maps api)
    _test_dataframe(df0)
    _test_column_in_dataframe(df0, address_tag)
    if city_tag is not None:
        _test_column_in_dataframe(df0, city_tag)
    if province_tag is not None:
        _test_column_in_dataframe(df0, province_tag)
    if regione_tag is not None:
        _test_column_in_dataframe(df0, regione_tag)

    col_list = [address_tag, city_tag, province_tag, regione_tag]
    col_list = [x for x in col_list if x is not None]
    df = df0[col_list].drop_duplicates()
    n = df.shape[0]
    df["address_search"] = df[address_tag].str.lower()
    if city_tag:
        t = __test_city_in_address(df, city_tag, "address_search")
        t = t | df[city_tag].isna()
        df["address_search"] = np.where(t, df["address_search"], df["address_search"] + ", " + df[city_tag].str.lower())

    log.info("Run search on OpenStreetMap. Needed at least {} seconds".format(n))
    geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    start = time.time()
    df["location"] = (df["address_search"]).apply(geocode)
    log.info("Finding locations from address ended in {} seconds".format(time.time() - start))
    df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
    df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)
    df["address_test"] = df["location"].apply(lambda loc: loc.address if loc else None).str.lower()
    df = df.assign(test=False)
    n_tot = df.shape[0]
    n_found = df["location"].notnull().sum()
    n_not_found = n_tot - n_found

    if n_not_found > 0:
        df = _try_replace_abbreviation_on_google(df, n_url_read, geocode)
        df = _try_wrong_replace_of_accents(df, address_tag, geocode)

    n_found = df["location"].notnull().sum()
    n_not_found = n_tot - n_found
    if city_tag:
        df["test"] = __test_city_in_address(df, city_tag, "address_test")
        df.loc[~df["test"], "latitude"] = None
        df.loc[~df["test"], "longitude"] = None
        log.info("Found {} location over {} address. But {} are not in the correct city.".format(n_found, n_tot, (~df["test"]).sum() - n_not_found))
    elif province_tag:
        df["test"] = df["test"] | __test_city_in_address(df, province_tag, "address_test")
        df.loc[~df["test"], "latitude"] = None
        df.loc[~df["test"], "longitude"] = None
        log.info("Found {} location over {} address. But {} are not in the correct provincia.".format(
            n_found, n_tot, (~df["test"]).sum() - n_not_found))
    elif regione_tag:
        df["test"] = df["test"] | __test_city_in_address(df, regione_tag, "address_test")
        df.loc[~df["test"], "latitude"] = None
        df.loc[~df["test"], "longitude"] = None
        log.info("Found {} location over {} address. But {} are not in the correct regione.".format(
            n_found, n_tot, (~df["test"]).sum() - n_not_found))
    else:
        log.info("Found {} location over {} address.".format(n_found, n_tot))

    # drop columns
    df.drop(["address_search", "location", "address_test", "test"], axis=1, inplace=True)
    ## Join df0
    df = df0.merge(df, how="left", on=col_list)
    return df


def get_address_from_coordinates(df0, latitude_columns=None, longitude_columns=None):
    _test_dataframe(df0)
    if latitude_columns is None or longitude_columns is None:
        flag_coord_found, latitude_columns, longitude_columns = __find_coord_columns(df0)
        if not flag_coord_found:
            raise Exception("Unable to find the latitude and longitude columns. Please specify them in latitude_columns and longitude_columns")
    try:
        df0[latitude_columns] = df0[latitude_columns].astype(float)
        df0[latitude_columns] = df0[latitude_columns].astype(float)
    except:
        raise Exception("Use columns with float type for coordinates.")
    if (latitude_columns is not None and latitude_columns not in df0.columns) or\
        (longitude_columns is not None and longitude_columns not in df0.columns):
        raise Exception("Use latitude_columns and longitude_column to specify the columns where to find the coordinates.")
    df = df0[[latitude_columns, longitude_columns]].drop_duplicates()
    df['geom'] = df[latitude_columns].map(str) + ', ' + df[longitude_columns].map(str)
    n = df.shape[0]
    log.info("Needed at least {} seconds".format(n))
    geolocator = Nominatim(timeout=10, user_agent=cfg.USER_AGENT)
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    start = time.time()
    df["location"] = (df["geom"]).apply(reverse)
    log.info("Finding address from coordinates ended in {} seconds".format(time.time() - start))

    address_col = "address"
    if address_col in df0.columns:
        address_col = "address_geo_ita"
    city_col = "city"
    if city_col in df0.columns:
        city_col = "city_geo_ita"
    # TODO Ripulire address estraendo solo informazioni utili e uniformi
    df[address_col] = df["location"].apply(lambda loc: loc.address if loc else None).str.lower()
    df[city_col] = df["location"].apply(lambda loc: loc.raw["address"]["city"] if (loc and "city" in loc.raw["address"]) else None).str.lower()
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
        dist = distance(a,b).m
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


def get_population_nearby2(df, radius, latitude_columns=None, longitude_columns=None):
    population_df = get_high_resolution_population_density_df()
    #population_df = __create_geo_dataframe(population_df)
    long_tag = "Lon"
    lat_tag = "Lat"
    df["key_mapping"] = range(df.shape[0])
    radius_df = __create_geo_dataframe(df, lat_tag=latitude_columns, long_tag=longitude_columns)[["key_mapping", "geometry"]]
    radius_df = radius_df.to_crs({'init': 'epsg:4326'})
    x_dist2, y_dist2 = _distance_to_range_ccord(radius*1.1)
    x_dist, y_dist = _distance_to_range_ccord(radius)
    dist = (x_dist + y_dist) / 2
    radius_df["geometry2"] = radius_df["geometry"]
    more_precision = False
    if more_precision:
        radius_df["geometry"] = radius_df.apply(lambda x: x['geometry2'].buffer(dist, cap_style=1), axis=1)
    radius_df["population0"] = None
    radius_df["population1"] = None
    start = time.time()
    print("Start")
    for index, row in radius_df.iterrows():
        signle_population_df = population_df[(population_df[lat_tag].between(row["geometry2"].y-y_dist2, row["geometry2"].y+y_dist2) &
                                              population_df[long_tag].between(row["geometry2"].x-x_dist2, row["geometry2"].x+x_dist2))]
        population = signle_population_df["Population"].sum()
        radius_df.loc[index, 'population0'] = population
        if more_precision & (population > 0):
            signle_population_df = gpd.GeoDataFrame(signle_population_df.drop([long_tag, lat_tag], axis=1),
                                            crs={'init': 'epsg:4326'},
                                            geometry=gpd.points_from_xy(signle_population_df[long_tag], signle_population_df[lat_tag]))
            circle = radius_df.loc[radius_df.index == index, ["geometry"]]
            signle_population_df = gpd.sjoin(signle_population_df, circle, op='within')
            radius_df.loc[index, 'population1'] = signle_population_df["Population"].sum()

    end = time.time()
    print('Change coord System', end - start)
    start = time.time()
    df["n_residents0"] = df["key_mapping"].map(radius_df.set_index("key_mapping")["population0"])
    if more_precision:
        df["n_residents1"] = df["key_mapping"].map(radius_df.set_index("key_mapping")["population1"])
    end = time.time()
    df.drop(["key_mapping"], axis=1, inplace=True)
    print('Mapping', end - start)
    return df


def get_population_nearby(df, radius, latitude_columns=None, longitude_columns=None):
    population_df = get_high_resolution_population_density_df()
    radius_df = df.rename_axis('key_mapping').reset_index()
    radius_df = __create_geo_dataframe(radius_df, lat_tag=latitude_columns, long_tag=longitude_columns)[["key_mapping", "geometry"]]
    radius_df = radius_df.to_crs({'init': 'epsg:4326'})
    if radius_df.shape[0] > 1000:
        log.info("Start creating the geopandas dataframe")
        start = datetime.now()
        long_tag = "Lon"
        lat_tag = "Lat"
        population_df = gpd.GeoDataFrame(
            population_df.drop([long_tag, lat_tag], axis=1),
            crs={'init': 'epsg:4326'},
            geometry=gpd.points_from_xy(population_df[long_tag], population_df[lat_tag]))
        end = datetime.now()
        log.info("Created the geopandas dataframe in {}".format(end - start))
        x_dist, y_dist = _distance_to_range_ccord(radius)
        dist = (x_dist + y_dist) / 2
        radius_df["geometry"] = radius_df.apply(lambda x: x['geometry'].buffer(dist, cap_style=1), axis=1)
        start = time.time()
        population_df = gpd.sjoin(population_df, radius_df, op='within')
        end = time.time()
        print('Join', end - start)
        start = time.time()
        mapping = population_df.groupby("key_mapping")["Population"].sum()
    else:
        pass
    df["n_residents"] = df["key_mapping"].map(mapping) / 10
    end = time.time()
    print('Mapping', end - start)
    return df


def _process_high_density_population_df(file_path, split_perc=0.05):
    file_path = str(file_path)
    df = pd.read_csv(file_path)

    #df = get_city_from_coordinates(df)
    #df.drop(["geometry"], axis=1, inplace=True)
    #df.to_pickle(file_path.replace(".csv", ".pkl"))

    # Split
    log.info("Start getting city from coordinates")
    split_path = PureWindowsPath(file_path.rsplit('\\', 1)[0]) / "split"
    os.mkdir(split_path)
    n_split = math.ceil(1 / split_perc)
    df_list = np.split(df, np.arange(1, n_split) * int(split_perc * len(df)))
    i = 0
    for df_i in df_list:
        df_i.to_pickle(split_path / PureWindowsPath(str(i) + ".pkl"))
        i += 1
    del df
    for filename in os.listdir(split_path):
        df = pd.read_pickle(split_path / filename)
        df = get_city_from_coordinates(df)
        df.to_pickle(split_path / filename)
    df = pd.DataFrame()
    for filename in os.listdir(split_path):
        df = pd.concat([pd.read_pickle(split_path / filename), df], ignore_index=True)
    df.drop(["geometry"], axis=1, inplace=True)
    log.info("End getting city from coordinates")
    df.to_pickle(str(file_path).replace(".csv", ".pkl"))


def aggregate_point_by_distance(df0, distance_in_meters, latitude_columns=None, longitude_columns=None, agg_column_name="aggregation_code"):
    _test_dataframe(df0)
    if latitude_columns is not None:
        _test_column_in_dataframe(df0, latitude_columns)
    if longitude_columns is not None:
        _test_column_in_dataframe(df0, longitude_columns)
    df0["key_mapping"] = range(df0.shape[0])
    df = __create_geo_dataframe(df0, latitude_columns, longitude_columns)
    df = df.to_crs({'init': 'epsg:4326'})
    df = df[["key_mapping", "geometry"]]
    radius_df = df.copy()
    #TODO Da rivedere
    x_dist, y_dist = _distance_to_range_ccord(distance_in_meters)
    dist = (x_dist + y_dist) / 2
    radius_df["geometry"] = radius_df.apply(lambda x: x['geometry'].buffer(dist, cap_style=1), axis=1)
    radius_df = gpd.sjoin(df, radius_df, op='within', how="left")
    radius_df = radius_df[["key_mapping_left", "key_mapping_right"]]
    radius_df.set_index(["key_mapping_left", "key_mapping_right"], inplace=True)
    radius_df["value"] = 1
    n_cc, df[agg_column_name] = connected_components(radius_df["value"].unstack().values)
    df = df.set_index("key_mapping")[agg_column_name]
    df0[agg_column_name] = df0["key_mapping"].map(df)
    df0.drop(["key_mapping"], axis=1, inplace=True)
    log.info("The {} points have been aggregated in {} group. The largest has {} points.".format(df0.shape[0], n_cc, df0[agg_column_name].value_counts().values[0]))
    return df0


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



