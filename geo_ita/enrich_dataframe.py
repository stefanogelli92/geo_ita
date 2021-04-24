import difflib
import time
import logging
import ssl

import numpy as np
import pandas as pd
import geopandas as gpd
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from geo_ita.data import get_df, create_df_comuni, get_variazioni_amministrative_df
from geo_ita.config import plot_italy_margins_4326, plot_italy_margins_32632
import geo_ita.config as cfg

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx


def add_geographic_info(df0, comuni_tag=None, province_tag=None, regioni_tag=None,
                        unique_flag=True, add_missing=False, drop_not_match=False):
    log.info("Add geo info: Start adding geographic info")

    df = df0.copy()

    if comuni_tag:
        level = cfg.LEVEL_COMUNE
        geo_tag_input = comuni_tag
    elif province_tag:
        level = cfg.LEVEL_PROVINCIA
        geo_tag_input = province_tag
    elif regioni_tag:
        geo_tag_input = regioni_tag
        level = cfg.LEVEL_REGIONE
    else:
        raise Exception("You need to pass al least one betweeen comuni_tag, province_tag or regioni_tag")

    code_type = _code_or_desc(list(df[geo_tag_input].unique()))

    geo_tag_anag = _get_tag_anag(code_type, level)

    log.debug("Add geo info: The most granular info is a {}".format(geo_tag_anag))

    info_df = get_df(level)
    info_df[cfg.KEY_UNIQUE] = info_df[geo_tag_anag]

    if (level == cfg.LEVEL_COMUNE) and (code_type == cfg.CODE_DENOMINAZIONE):
        log.debug("Add geo info: The dataset could contains an homonym comune")
        info_tag_details = cfg.TAG_SIGLA
        if province_tag is not None:
            code_province = _code_or_desc(list(df[province_tag].unique()))
            geo_tag_anag2 = _get_tag_anag(code_province, cfg.LEVEL_PROVINCIA)
            log.debug("Add geo info: The user has specified the {} so we can use in order to correctly match homonym comuni".format(geo_tag_anag2))
            info_tag_details = geo_tag_anag2
            split_denom_comuni_omonimi(df, geo_tag_input, province_tag, geo_tag_anag2)
        elif regioni_tag is not None:
            code_regioni = _code_or_desc(list(df[regioni_tag].unique()))
            geo_tag_anag2 = _get_tag_anag(code_regioni, cfg.LEVEL_REGIONE)
            log.debug(
                "Add geo info: The user has specified the {} so we can use in order to correctly match homonym comuni".format(
                    geo_tag_anag2))
            info_tag_details = geo_tag_anag2
            split_denom_comuni_omonimi(df, geo_tag_input, regioni_tag, geo_tag_anag2)
        split_denom_comuni_omonimi(info_df, cfg.KEY_UNIQUE, info_tag_details, info_tag_details)

    df, info_df = __uniform_names(df, info_df,
                                  geo_tag_input,
                                  cfg.KEY_UNIQUE,
                                  cfg.KEY_UNIQUE,
                                  unique_flag=unique_flag,
                                  comune_flag=(level == cfg.LEVEL_COMUNE))

    if add_missing:
        if drop_not_match:
            how = "left"
        else:
            how = "outer"
        df = info_df.merge(df, on=cfg.KEY_UNIQUE, how=how)
    else:
        if drop_not_match:
            how = "inner"
        else:
            how = "left"
        df = df.merge(info_df,  on=cfg.KEY_UNIQUE, how=how)

    drop_col = [cfg.KEY_UNIQUE, geo_tag_input]
    for col in drop_col:
        if col in df.columns:
            df.drop([col], axis=1, inplace=True)
    return df


def split_denom_comuni_omonimi(df, den_tag, details_tag1, details_tag2):
    split_df = pd.DataFrame.from_dict(cfg.comuni_omonimi)
    split_df[cfg.TAG_COMUNE] = split_df[cfg.TAG_COMUNE].str.lower()
    split_df[details_tag2] = split_df[details_tag2].astype(str).str.lower()
    split_df["key"] = split_df[cfg.TAG_COMUNE] + split_df[details_tag2]
    pos = df[den_tag].str.lower().isin(split_df[cfg.TAG_COMUNE].unique())
    df[den_tag] = np.where(pos, df[den_tag].str.lower() + df[details_tag1].astype(str).str.lower(),
                                 df[den_tag])
    df[den_tag] = df[den_tag].replace(split_df.set_index("key")["new_name"])
    return df


def _code_or_desc(list_values):
    list_values = [x for x in list_values if str(x) != 'nan']
    n_tot = len(list_values)
    if (sum([isinstance(item, int) or item.isdigit() for item in list_values]) / n_tot) > 0.8:
        result = cfg.CODE_CODICE_ISTAT
    elif (sum([isinstance(item, str) and item.isalpha() and len(item) == 2 for item in list_values]) / n_tot) > 0.8:
        result = cfg.CODE_SIGLA
    else:
        result = cfg.CODE_DENOMINAZIONE
    return result


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
                df0, geometry=gpd.points_from_xy(df0[lat_tag], df0[long_tag]))
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
    # TODO Controllo correttezza margini italia
    center_lat = df[lat].median()
    center_lon = df[lon].median()

    if (plot_italy_margins_4326[0][0] <= center_lat <= plot_italy_margins_4326[0][1]) & \
            (plot_italy_margins_4326[1][0] <= center_lon <= plot_italy_margins_4326[1][1]):
        result = "epsg:4326"
        log.info("Found coord system: {}".format(result))
    elif (plot_italy_margins_32632[0][0] <= center_lat <= plot_italy_margins_32632[0][1]) & \
            (plot_italy_margins_32632[1][0] <= center_lon <= plot_italy_margins_32632[1][1]):
        result = "epsg:32632"
        log.info("Found coord system: {}".format(result))
    else:
        result = "epsg:32632"
        log.warning("Unable to find coord system so the default is used: {}".format(result))
    return result


def get_city_from_coordinates(df0, rename_col_comune=None, rename_col_provincia=None, rename_col_regione=None):
    # TODO GET comune - provincia - regione
    df0 = df0.rename_axis('key_mapping').reset_index()
    df_comuni = create_df_comuni()
    df_comuni = gpd.GeoDataFrame(df_comuni)
    df_comuni.crs = {'init': 'epsg:32632'}
    df_comuni = df_comuni.to_crs({'init': 'epsg:4326'})

    df = __create_geo_dataframe(df0)
    df = df[["key_mapping", "geometry"]].drop_duplicates()
    df = df.to_crs({'init': 'epsg:4326'})

    map_city = gpd.sjoin(df, df_comuni, op='within')
    n_tot = map_city.shape[0]
    missing = list(map_city[map_city[cfg.TAG_COMUNE].isna()]["geometry"].unique())
    if len(missing) == n_tot:
        log.info("Found the correct city for each point")
    else:
        log.warning("Unable to find the city for {} points: {}".format(len(missing), missing))
    map_city = map_city[["key_mapping", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE]]
    rename_col = {}
    if rename_col_comune is not None:
        rename_col[cfg.TAG_COMUNE] = rename_col_comune
    if rename_col_provincia is not None:
        rename_col[cfg.TAG_PROVINCIA] = rename_col_provincia
    if rename_col_regione is not None:
        rename_col[cfg.TAG_REGIONE] = rename_col_regione
    map_city.rename(columns=rename_col, inplace=True)
    return df0.merge(map_city, on=["key_mapping"], how="left").drop(["key_mapping"], axis=1)


def __clean_denom_text(series):
    series = series.str.lower()  # All strig in lowercase
    series = series.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non alphabetic characters
    series = series.str.strip()
    series = series.str.replace(r'\s+', ' ', regex=True)
    series = series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8') # Remove accent
    return series


def __find_match(not_match1, not_match2, unique=False):
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
                match_dict[a] = best_match
                if unique:
                    not_match2.remove(best_match)
    return match_dict


def __uniform_names(df1, df2, tag_1, tag_2, tag, unique_flag=True, comune_flag=False):
    # TODO add split (/)
    """
    Parameters
    ----------
    df1: dataset Principale
    df2: dataset Secondario usato come anagrafica
    tag_1: nome della colonna del dataset df1 contenente i nomi da uniformare
    tag_2: nome della colonna del dataset df2 contenente i nomi da uniformare
    tag: Tag di output dei nomi uniformati

    Returns
    Restituisce i dataset di partenza con l'aggiunta della colonna uniformata, l'elenco dei valori non abbinati,
    l'elenco dei valori del dataset2 che no nsono stati utilizzati, un dizionario contenente la mappatura dei valori
    non abbinati al vaore più simile.
    """
    if tag_1 == tag:
        df1[tag_1 + "_original"] = df1[tag_1]
    if tag_2 == tag:
        df2[tag_2 + "_original"] = df2[tag_2]
    if comune_flag:
        df1[tag_1] = df1[tag_1].str.lower().replace(cfg.comuni_exceptions)
        df2[tag_2] = df2[tag_2].str.lower().replace(cfg.comuni_exceptions)
    df1[tag] = __clean_denom_text(df1[tag_1])
    df2[tag] = __clean_denom_text(df2[tag_2])
    if comune_flag:
        df1[tag] = df1[tag].replace(cfg.comuni_exceptions)
        df2[tag] = df2[tag].replace(cfg.comuni_exceptions)
        # Replace found manually
        df1[tag] = df1[tag].replace(cfg.rename_comuni_nomi)
        # replace variazioni amministrative nella storia
        df_variazioni = get_variazioni_amministrative_df()
        df_variazioni = df_variazioni[df_variazioni["tipo_variazione"].isin(["ES", "CD"])]
        df_variazioni[cfg.TAG_COMUNE] = __clean_denom_text(df_variazioni[cfg.TAG_COMUNE])
        df_variazioni["new_denominazione_comune"] = __clean_denom_text(df_variazioni["new_denominazione_comune"])
        df_variazioni["data_decorrenza"] = pd.to_datetime(df_variazioni["data_decorrenza"])
        df_variazioni.sort_values([cfg.TAG_COMUNE, "data_decorrenza"], ascending=False, inplace=True)
        df_variazioni = df_variazioni.groupby(cfg.TAG_COMUNE)["new_denominazione_comune"].last().to_dict()
        log.debug(df_variazioni)
        log.debug(df1[tag])
        df1[tag] = df1[tag].replace(df_variazioni)

    den1 = df1[df1[tag].notnull()][tag].unique()
    den2 = df2[df2[tag].notnull()][tag].unique()
    not_match1 = list(set(den1) - set(den2))
    if unique_flag:
        not_match2 = list(set(den2) - set(den1))
        match_dict = __find_match(not_match2, not_match1, unique=True)
        match_dict = {v: k for k, v in match_dict.items()}
        not_match1 = [x for x in not_match1 if x not in match_dict.keys()]
    else:
        not_match2 = den2
        match_dict = __find_match(not_match1, not_match2)

    not_match1 = [x for x in not_match1 if x not in match_dict.keys()]
    df1[tag] = df1[tag].replace(match_dict)

    if len(not_match1) == 0:
        log.info("All name have been matched")
    else:
        log.warning("Unable to match {} names: {}".format(len(not_match1), not_match1))
    return df1, df2


def __test_city_in_address(df, city_tag, address_tag):
    return df.apply(lambda x: x[city_tag].lower() in x[address_tag] if x[address_tag] else False, axis=1)


def get_coordinates_from_address(df, address_tag, city_tag=None, province_tag=None, regione_tag=None):
    # TODO log
    # TODO add successive tentative (maps api)

    df["address_search"] = df[address_tag].str.lower()
    if city_tag:
        df["test"] = __test_city_in_address(df, city_tag, "address_search")
        perc_success = df["test"].sum() / df.shape[0]
        if perc_success < 0.1:
            df["address_search"] = df["address_search"] + ", " + df[city_tag].str.lower()
    geolocator = Nominatim(user_agent="trial")  # "trial"
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    start = time.time()
    df["location"] = (df["address_search"]).apply(geocode)
    log.info("Finding of location from address ended in {} seconds".format(time.time() - start))
    df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
    df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)
    df["address_test"] = df["location"].apply(lambda loc: loc.address if loc else None).str.lower()
    df["test"] = False
    if city_tag:
        df["test"] = __test_city_in_address(df, city_tag, "address_test")
        log.info("Found {} location over {} address. But {} are not in the correct city.".format(df["location"].notnull().sum(), df.shape[0], (~df["test"]).sum()))
    elif province_tag:
        df["test"] = df["test"] | __test_city_in_address(df, province_tag, "address_test")
        log.info("Found {} location over {} address. But {} are not in the correct provincia.".format(
            df["location"].notnull().sum(), df.shape[0], (~df["test"]).sum()))
    elif regione_tag:
        df["test"] = df["test"] | __test_city_in_address(df, regione_tag, "address_test")
        log.info("Found {} location over {} address. But {} are not in the correct regione.".format(
            df["location"].notnull().sum(), df.shape[0], (~df["test"]).sum()))
    else:
        df.drop(["test"], axis=1, inplace=True)
        log.info("Found {} location over {} address.".format(df["location"].notnull().sum(), df.shape[0]))
    return df
