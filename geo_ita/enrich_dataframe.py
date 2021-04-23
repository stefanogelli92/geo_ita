import difflib
import time
import logging
import ssl

import pandas as pd
import geopandas as gpd
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from geo_ita.data import create_df_comuni, get_anagrafica_df
from geo_ita.config import plot_italy_margins_4326, plot_italy_margins_32632
import geo_ita.config as cfg

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx


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
        log.debug("Found coord system: {}".format(result))
    elif (plot_italy_margins_32632[0][0] <= center_lat <= plot_italy_margins_32632[0][1]) & \
            (plot_italy_margins_32632[1][0] <= center_lon <= plot_italy_margins_32632[1][1]):
        result = "epsg:32632"
        log.debug("Found coord system: {}".format(result))
    else:
        result = "epsg:32632"
        log.warning("Unable to find coord system so the default is used: {}".format(result))
    return result


def get_city_from_coordinates(df0, comune_tag=None, provincia_tag=None, regione_tag=None):
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
    missing = map_city[cfg.TAG_COMUNE].isna().sum()
    if len(missing) == n_tot:
        log.debug("Found the correct city for each point")
    else:
        log.warning("Unable to find the city for {} points: {}".format(len(missing), missing))
    map_city = map_city[["key_mapping", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE]]
    rename_col = {}
    if comune_tag is not None:
        rename_col[cfg.TAG_COMUNE] = comune_tag
    if provincia_tag is not None:
        rename_col[cfg.TAG_PROVINCIA] = provincia_tag
    if regione_tag is not None:
        rename_col[cfg.TAG_REGIONE] = regione_tag
    map_city.rename(columns=rename_col, inplace=True)
    return df0.merge(map_city, on=["key_mapping"], how="left").drop(["key_mapping"], axis=1)


def __clean_denom_text(series):
    series = series.str.lower()  # All strig in lowercase
    series = series.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non alphabetic characters
    series = series.str.strip()
    series = series.str.replace(r'\s+', ' ', regex=True)
    return series


def __find_match(not_match1, not_match2):
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
        w_best = ""
        p_best = 0
        for b in not_match2:
            p = difflib.SequenceMatcher(None, a, b).ratio()
            if p > p_best:
                p_best = p
                w_best = b
            match_dict[a] = (w_best, p_best)
    return match_dict


def __uniform_names(df1, df2, tag_1, tag_2, tag, unique_flag=True):
    # TODO add split (/)
    # TODO add replace city - province - ita eng
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
    df1[tag] = __clean_denom_text(df1[tag_1])
    df2[tag] = __clean_denom_text(df2[tag_2])
    df1[tag] = df1[tag_1].replace(cfg.rename_comuni_nomi)
    den1 = df1[df1[tag].notnull()][tag].unique()
    den2 = df2[df2[tag].notnull()][tag].unique()
    not_match1 = list(set(den1) - set(den2))
    if unique_flag:
        not_match2 = list(set(den2) - set(den1))
    else:
        not_match2 = den2
    match_dict = __find_match(not_match1, not_match2)
    items = [(k, v[0], v[1]) for k, v in match_dict.items()]
    items = sorted(items, key=lambda tup: (tup[1], tup[2]), reverse=True)
    for a, b, c in items:
        if (b in not_match2) & (c >= cfg.min_acceptable_similarity):
            df1[tag] = df1[tag].replace(a, b)
            not_match1.remove(a)
            if unique_flag:
                not_match2.remove(b)
    if len(not_match1) == 0:
        log.debug("All name have been matched")
    else:
        log.warning("Unable to match {} names: {}".format(len(not_match1), not_match1))
    return df1, df2


def get_province_from_city(df0, comuni_tag, unique_flag=True):
    # TODO log
    df = df0.copy()
    anagrafica = get_anagrafica_df()
    df, anagrafica = __uniform_names(df, anagrafica,
                                     comuni_tag, cfg.TAG_COMUNE, cfg.TAG_COMUNE,
                                     unique_flag=unique_flag)
    df = df.merge(
        anagrafica[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE]],
        on=cfg.TAG_COMUNE, how="left")
    return df


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
        log.debug("Found {} location over {} address. But {} are not in the correct city.".format(df["location"].notnull().sum(), df.shape[0], (~df["test"]).sum()))
    elif province_tag:
        df["test"] = df["test"] | __test_city_in_address(df, province_tag, "address_test")
        log.debug("Found {} location over {} address. But {} are not in the correct provincia.".format(
            df["location"].notnull().sum(), df.shape[0], (~df["test"]).sum()))
    elif regione_tag:
        df["test"] = df["test"] | __test_city_in_address(df, regione_tag, "address_test")
        log.debug("Found {} location over {} address. But {} are not in the correct regione.".format(
            df["location"].notnull().sum(), df.shape[0], (~df["test"]).sum()))
    else:
        df.drop(["test"], axis=1, inplace=True)
        log.debug("Found {} location over {} address.".format(df["location"].notnull().sum(), df.shape[0]))
    return df
