import difflib
import re
import time
import logging
import ssl
import unidecode

import numpy as np
import pandas as pd
import geopandas as gpd
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
# from sklearn.neighbors import KernelDensity

from geo_ita.src._data import get_df, get_df_comuni, get_variazioni_amministrative_df, _get_list, \
    get_double_languages_mapping, _get_shape_italia
import geo_ita.src.config as cfg

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx


class AddGeographicalInfo:

    def __init__(self, df):
        self.original_df = df
        self.keys = None
        self.df = None
        self.comuni, self.province, self.sigle, self.regioni = _get_list()
        self._find_info()
        self.cap_tag = None
        self.cap_code = None
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
        self.comuni_tag = col_name
        self.comuni_code = _code_or_desc(list(self.original_df[col_name].unique()))
        self.level = cfg.LEVEL_COMUNE
        self.geo_tag_input = col_name
        self.code = self.comuni_code

    # def set_cap_tag(self, col_name):
    #     list_values = [x for x in list(self.original_df[col_name].unique()) if (str(x) != 'nan') and (x is not None)]
    #     wrong_format = [x for x in list_values if (not isinstance(x, int)) & (not x.isdigit())]
    #     if len(wrong_format) == 0:
    #         self.cap_tag = col_name
    #         self.comuni_code = cfg.CODE_CAP
    #         if self.level != cfg.LEVEL_COMUNE:
    #             self.level = cfg.LEVEL_CAP
    #             self.geo_tag_input = col_name
    #             self.code = self.comuni_code
    #     else:
    #         raise Exception("{} values with a wrong format: {}".format(len(wrong_format), wrong_format))

    def set_province_tag(self, col_name):
        self.province_tag = col_name
        self.province_code = _code_or_desc(list(self.original_df[col_name].unique()))
        if (self.level != cfg.LEVEL_COMUNE) & (self.level != cfg.LEVEL_CAP):
            self.level = cfg.LEVEL_PROVINCIA
            self.geo_tag_input = col_name
            self.code = self.province_code

    def set_regioni_tag(self, col_name):
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

        self.keys = [x for x in [self.cap_tag, self.comuni_tag, self.province_tag, self.regioni_tag] if x is not None]
        if len(self.keys) == 0:
            raise Exception("You need to set al least one betweeen cap_tag, comuni_tag, province_tag or regioni_tag.")
        self.df = self.original_df[self.keys].copy().drop_duplicates()

        self.df = self.df[self.df[self.geo_tag_input].notnull()]

        if self.level == cfg.LEVEL_CAP:
            self._map_cap_to_comune()

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

    def _map_cap_to_comune(self):
        self.df = self.df[self.df[self.geo_tag_input].astype(int) != 0]
        self.df[cfg.KEY_UNIQUE2] = self.df[self.geo_tag_input].apply(lambda x: x.zfill(5))
        #self.df[cfg.KEY_UNIQUE2] = self.df[cfg.KEY_UNIQUE2].map(xxx)
        self.level = cfg.LEVEL_COMUNE
        self.geo_tag_input = cfg.KEY_UNIQUE2
        self.code = cfg.CODE_DENOMINAZIONE
        self.df = self.df[self.df[self.geo_tag_input].notnull()]

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
        s = self.df[cfg.KEY_UNIQUE].replace(df_variazioni)
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
        geolocator = Nominatim(user_agent="trial")  # "trial"
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        regioni = [_clean_denom_text_value(a) for a in self.regioni]
        province = [_clean_denom_text_value(a) for a in self.province]
        comuni = [_clean_denom_text_value(a) for a in self.comuni]
        match_dict = {}
        for el in self.not_match:
            p = geocode(el + ", italia")
            if p is not None:
                address = p.address
                extract = re.search(
                    '(?P<comune2>[^,]+, )?(?P<comune1>[^,]+), (?P<provincia>[^,]+), (?P<regione>[^,0-9]+)(?P<cap>, [0-9]{5})?, Italia',
                    address)
                if extract and not any(word.lower() in address.lower() for word in ["via", "viale", "piazza"]):
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

    def get_result_frazioni(self):
        return self.frazioni_dict

    def get_result(self, add_missing=False, drop_not_match=False):
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
    if (sum([isinstance(item, np.integer) or item.isdigit() for item in list_values]) / n_tot) > 0.8:
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
    return flag_coord_found, lat_tag, long_tag


def __create_geo_dataframe(df0):
    if isinstance(df0, pd.DataFrame):
        flag_coord_found, lat_tag, long_tag = __find_coord_columns(df0)
        if flag_coord_found:
            df = gpd.GeoDataFrame(
                df0, geometry=gpd.points_from_xy(df0[long_tag], df0[lat_tag]))
            coord_system = __find_coordinates_system(df0, lat_tag, long_tag)
            df.crs = {'init': coord_system}
            log.info("Found columns about coordinates: ({}, {})".format(lat_tag, long_tag))
        elif "geometry" in df0.columns:
            df = gpd.GeoDataFrame(df0)
            log.info("Found geometry columns")
        else:
            raise Exception("The DataFrame must have a geometry attribute or lat-long.")
    elif isinstance(df0, gpd.GeoDataFrame):
        df = df0.copy()
    else:
        raise Exception("You need to pass a Pandas DataFrame of GeoDataFrame.")
    return df


def __find_coordinates_system(df, lat, lon):
    n_test = min(100, df.shape[0])
    test = df.sample(n=n_test)
    test = gpd.GeoDataFrame(
        test, geometry=gpd.points_from_xy(test[lon], test[lat]))

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


def get_city_from_coordinates(df0, rename_col_comune=None, rename_col_provincia=None, rename_col_regione=None):
    df0 = df0.rename_axis('key_mapping').reset_index()
    df = df0.copy()
    df_comuni = get_df_comuni()
    df_comuni = gpd.GeoDataFrame(df_comuni)
    df_comuni.crs = {'init': 'epsg:32632'}
    df_comuni = df_comuni.to_crs({'init': 'epsg:4326'})

    df = __create_geo_dataframe(df)
    df = df[["key_mapping", "geometry"]].drop_duplicates()
    df = df.to_crs({'init': 'epsg:4326'})

    n_tot = df.shape[0]
    map_city = gpd.sjoin(df, df_comuni, op='within')

    missing = list(map_city[map_city[cfg.TAG_COMUNE].isna()]["geometry"].unique())
    if len(missing) == 0:
        log.info("Found the correct city for each point")
    else:
        log.warning("Unable to find the city for {} points: {}".format(len(missing), missing))
    map_city = map_city[["key_mapping", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]]
    rename_col = {}
    if rename_col_comune is not None:
        rename_col[cfg.TAG_COMUNE] = rename_col_comune
    if rename_col_provincia is not None:
        rename_col[cfg.TAG_PROVINCIA] = rename_col_provincia
    if rename_col_regione is not None:
        rename_col[cfg.TAG_REGIONE] = rename_col_regione
    map_city.rename(columns=rename_col, inplace=True)
    return df0.merge(map_city, on=["key_mapping"], how="left").drop(["key_mapping"], axis=1)


def __test_city_in_address(df, city_tag, address_tag):
    return df.apply(lambda x: x[city_tag].lower() in x[address_tag] if x[address_tag] else False, axis=1)


def get_coordinates_from_address(df0, address_tag, city_tag=None, province_tag=None, regione_tag=None):
    # TODO add successive tentative (maps api)
    col_list = [address_tag, city_tag, province_tag, regione_tag]
    col_list = [x for x in col_list if x is not None]
    df = df0[col_list].drop_duplicates()
    n = df.shape[0]
    df["address_search"] = df[address_tag].str.lower()
    if city_tag:
        t = __test_city_in_address(df, city_tag, "address_search")
        df["address_search"] = np.where(t, df["address_search"], df["address_search"] + ", " + df[city_tag].str.lower())

    log.info("Needed at least {} seconds".format(n))
    geolocator = Nominatim(user_agent="geo_ita")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    start = time.time()
    df["location"] = (df["address_search"]).apply(geocode)
    log.info("Finding of location from address ended in {} seconds".format(time.time() - start))
    df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
    df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)
    df["address_test"] = df["location"].apply(lambda loc: loc.address if loc else None).str.lower()
    df["test"] = False
    n_tot = df.shape[0]
    n_found = df["location"].notnull().sum()
    n_not_found = n_tot - n_found
    if city_tag:
        df["test"] = __test_city_in_address(df, city_tag, "address_test")
        log.info("Found {} location over {} address. But {} are not in the correct city.".format(n_found, n_tot, (~df["test"]).sum() - n_not_found))
    elif province_tag:
        df["test"] = df["test"] | __test_city_in_address(df, province_tag, "address_test")
        log.info("Found {} location over {} address. But {} are not in the correct provincia.".format(
            n_found, n_tot, (~df["test"]).sum() - n_not_found))
    elif regione_tag:
        df["test"] = df["test"] | __test_city_in_address(df, regione_tag, "address_test")
        log.info("Found {} location over {} address. But {} are not in the correct regione.".format(
            n_found, n_tot, (~df["test"]).sum() - n_not_found))
    else:
        df.drop(["test"], axis=1, inplace=True)
        log.info("Found {} location over {} address.".format(n_found, n_tot))

    df.loc[~df["test"], "latitude"] = None
    df.loc[~df["test"], "longitude"] = None
    # drop columns
    df.drop(["address_search", "location", "address_test", "test"], axis=1, inplace=True)
    ## Join df0
    df = df0.merge(df, how="left", on=col_list)
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



