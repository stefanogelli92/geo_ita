import difflib
import time

import pandas as pd
import geopandas as gpd
from geo_ita.data import create_df_comuni, get_anagrafica_df
from geo_ita.config import plot_italy_margins_4326, plot_italy_margins_32632
import geo_ita.config as cfg

import ssl
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
geopy.geocoders.options.default_ssl_context = ctx


def _find_coord_columns(df):
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


def _create_geo_dataframe(df0):
    if isinstance(df0, pd.DataFrame):
        flag_coord_found, lat_tag, long_tag = _find_coord_columns(df0)
        if flag_coord_found:
            df = gpd.GeoDataFrame(
                df0, geometry=gpd.points_from_xy(df0[lat_tag], df0[long_tag]))
            coord_system = _find_coordinates_system(df0, lat_tag, long_tag)
            df.crs = {'init': coord_system}
        elif "geometry" in df0.columns:
            df = gpd.GeoDataFrame(df0)
        else:
            raise Exception("The DataFrame must have a geometry attribute or lat-long.")
    elif isinstance(df0, gpd.GeoDataFrame):
        df = df0.copy()
    else:
        raise Exception("You need to pass a Pandas DataFrame of GeoDataFrame.")
    return df


def _find_coordinates_system(df, lat, lon):
    # TODO Controllo correttezza margini italia
    center_lat = df[lat].median()
    center_lon = df[lon].median()

    if (plot_italy_margins_4326[0][0] <= center_lat <= plot_italy_margins_4326[0][1]) & \
        (plot_italy_margins_4326[1][0] <= center_lon <= plot_italy_margins_4326[1][1]):
        result = "epsg:4326"
    elif (plot_italy_margins_32632[0][0] <= center_lat <= plot_italy_margins_32632[0][1]) & \
        (plot_italy_margins_32632[1][0] <= center_lon <= plot_italy_margins_32632[1][1]):
        result = "epsg:32632"
    else:
        result = "epsg:32632"
    return result


def get_city_from_coordinates(df0, comune_tag=None, provincia_tag=None, regione_tag=None):
    # TODO GET comune - provincia - regione
    df0 = df0.rename_axis('key_mapping').reset_index()
    df_comuni = create_df_comuni()
    df_comuni = gpd.GeoDataFrame(df_comuni)
    df_comuni.crs = {'init': 'epsg:32632'}
    df_comuni = df_comuni.to_crs({'init': 'epsg:4326'})

    df = _create_geo_dataframe(df0)
    df = df[["key_mapping", "geometry"]].drop_duplicates()
    df = df.to_crs({'init': 'epsg:4326'})

    map_city = gpd.sjoin(df, df_comuni, op='within')
    map_city = map_city[["key_mapping", "denominazione_comune", "denominazione_provincia", "Regione"]]
    rename_col = {}
    if comune_tag is not None:
        rename_col["denominazione_comune"] = comune_tag
    if provincia_tag is not None:
        rename_col["denominazione_provincia"] = provincia_tag
    if regione_tag is not None:
        rename_col["Regione"] = regione_tag
    map_city.rename(columns=rename_col, inplace=True)
    return df0.merge(map_city, on=["key_mapping"], how="left").drop(["key_mapping"], axis=1)


def _clean_denom_text(series):
    series = series.str.lower()  # All strig in lowercase
    series = series.str.replace('[^\w\s]', ' ')  # Remove non alphabetic characters
    series = series.str.strip()
    series = series.str.replace('\s+', ' ')
    return series


def _find_match(not_match1, not_match2):
    """
    Parameters
    ----------
    not_match1: Lista di nomi da abbinare ad un valore della lista not_match2
    not_match2: Lista dei nomi a cui abbinare un valore della lista not_match1

    Returns
    Restituisce un dizionario contenente per ogni parole di not_match1 la parola più simile di not_match2 con il
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


def _uniform_names(df1, df2, tag_1, tag_2, tag, unique_flag=True):
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
    df1[tag] = _clean_denom_text(df1[tag_1])
    df2[tag] = _clean_denom_text(df2[tag_2])
    den1 = df1[df1[tag].notnull()][tag].unique()
    den2 = df2[df2[tag].notnull()][tag].unique()
    not_match1 = list(set(den1) - set(den2))
    if unique_flag:
        not_match2 = list(set(den2) - set(den1))
    else:
        not_match2 = den2
    match_dict = _find_match(not_match1, not_match2)
    items = [(k, v[0], v[1]) for k, v in match_dict.items()]
    items = sorted(items, key=lambda tup: (tup[1], tup[2]), reverse=True)
    for a, b, c in items:
        if (b in not_match2) & (c >= cfg.min_acceptable_similarity):
            df1[tag] = df1[tag].replace(a, b)
            not_match1.remove(a)
            if unique_flag:
                not_match2.remove(b)
    print("{} unknown comuni".format(len(not_match1)))
    return df1, df2


def get_province_from_city(df0, comuni_tag, unique_flag=True):
    # TODO log
    df = df0.copy()
    anagrafica = get_anagrafica_df()
    df, anagrafica = _uniform_names(df, anagrafica, comuni_tag, "denominazione_comune",
                                                 "denominazione_comune", unique_flag=unique_flag)
    df = df.merge(
        anagrafica[["denominazione_comune", "denominazione_provincia", "codice_provincia"]],
        on="denominazione_comune", how="left")
    return df


def _test_city_in_address(df, city_tag, address_tag):
    return df.apply(lambda x: x[city_tag].lower() in x[address_tag] if x[address_tag] else False, axis=1)


def get_coordinates_from_address(df, address_tag, city_tag=None, province_tag=None, regione_tag=None):
    # TODO log
    # TODO add successive tentative (maps api)

    df["address_search"] = df[address_tag].str.lower()
    if city_tag:
        df["test"] = _test_city_in_address(df, city_tag, "address_search")
        perc_success = df["test"].sum() / df.shape[0]
        if perc_success < 0.1:
            df["address_search"] = df["address_search"] + ", " + df[city_tag].str.lower()
    geolocator = Nominatim(user_agent="trial")  # "trial"
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    start = time.time()
    df["location"] = (df["address_search"]).apply(geocode)
    print(time.time() - start)
    df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
    df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)
    df["address_test"] = df["location"].apply(lambda loc: loc.address if loc else None).str.lower()
    df["test"] = False
    if city_tag:
        df["test"] = _test_city_in_address(df, city_tag, "address_test")
    elif province_tag:
        df["test"] = df["test"] | _test_city_in_address(df, province_tag, "address_test")
    elif regione_tag:
        df["test"] = df["test"] | _test_city_in_address(df, regione_tag, "address_test")
    return df



