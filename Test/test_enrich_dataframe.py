from geo_ita.src._enrich_dataframe import __find_coordinates_system

from geo_ita.src._enrich_dataframe import *
from geo_ita.src._data import get_df_comuni
from geo_ita.src.definition import *
from pathlib import PureWindowsPath
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def test_add_geographic_info():
    path = root_path / PureWindowsPath(r"data_sources/Test/uniform_name.xlsx")
    test_df = pd.read_excel(path)
    n_tot = test_df.shape[0]

    geoInf = AddGeographicalInfo(test_df)
    geoInf.set_comuni_tag("Citta")
    geoInf.run_simple_match()
    geoInf.run_find_frazioni()
    #geoInf.run_similarity_match(unique_flag=False)
    #geoInf.accept_similarity_result()
    result = geoInf.get_result()

    n_right = (result[cfg.TAG_COMUNE].fillna('-').eq(result["comune"].fillna('-'))).sum()
    n_right2 = (result[cfg.TAG_PROVINCIA].fillna('-').eq(result["provincia"].fillna('-'))).sum()
    perc_success = n_right / n_tot
    log.info("Test1: {}, {}".format(round(perc_success * 100, 1), (n_right2 == n_tot - 1)))
    if perc_success != 1:
        log.warning("\n" + result.to_string())

    geoInf = AddGeographicalInfo(test_df)
    geoInf.set_comuni_tag("Citta")
    geoInf.set_province_tag("sl")
    geoInf.run_simple_match()
    geoInf.run_find_frazioni()
    geoInf.run_similarity_match(unique_flag=False)
    geoInf.accept_similarity_result()
    result = geoInf.get_result()

    n_right = ((result[cfg.TAG_COMUNE].fillna('-').eq(result["comune"].fillna('-'))) &
                (result[cfg.TAG_PROVINCIA].fillna('-').eq(result["provincia"].fillna('-')))).sum()
    perc_success = n_right / n_tot
    log.info("Test2:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + result.to_string())

    geoInf = AddGeographicalInfo(test_df)
    geoInf.set_comuni_tag("Citta")
    geoInf.set_province_tag("provincia")
    geoInf.run_simple_match()
    geoInf.run_find_frazioni()
    geoInf.run_similarity_match(unique_flag=False)
    geoInf.accept_similarity_result()
    result = geoInf.get_result()

    n_right = ((result[cfg.TAG_COMUNE].fillna('-').eq(result["comune"].fillna('-'))) &
               (result[cfg.TAG_PROVINCIA].fillna('-').eq(result["provincia"].fillna('-')))).sum()
    perc_success = n_right / n_tot
    log.info("Test3:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + result.to_string())

    geoInf = AddGeographicalInfo(test_df)
    geoInf.set_comuni_tag("Citta")
    geoInf.set_regioni_tag("regione")
    geoInf.run_simple_match()
    geoInf.run_find_frazioni()
    geoInf.run_similarity_match(unique_flag=False)
    geoInf.accept_similarity_result()
    result = geoInf.get_result()

    n_right = ((result[cfg.TAG_COMUNE].fillna('-').eq(result["comune"].fillna('-'))) &
               (result[cfg.TAG_PROVINCIA].fillna('-').eq(result["provincia"].fillna('-')))).sum()
    perc_success = n_right / n_tot
    log.info("Test4:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + result.to_string())

    geoInf = AddGeographicalInfo(test_df)
    geoInf.set_province_tag("sl")
    geoInf.set_regioni_tag("regione")
    geoInf.run_simple_match()
    result = geoInf.get_result()

    n_right = (result[cfg.TAG_PROVINCIA].fillna('-').eq(result["provincia"].fillna('-')) |
               result["provincia"].isna() | result["sl"].isna()).sum()
    perc_success = n_right / n_tot
    log.info("Test5:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + result.to_string())

    geoInf = AddGeographicalInfo(test_df)
    geoInf.set_province_tag("provincia")
    geoInf.run_simple_match()
    result = geoInf.get_result()

    n_right = (result[cfg.TAG_SIGLA].fillna('-').eq(result["sl"].fillna('-')) |
               result["provincia"].isna() | result["sl"].isna()).sum()
    perc_success = n_right / n_tot
    log.info("Test6:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + result.to_string())


def test_get_city_from_coordinates():
    test_df = get_df_comuni()
    test_df.rename(columns={"center_x": "lon",
                            "center_y": "lat"}, inplace=True)
    #test_df = pd.read_pickle(r"C:\Users\A470222\Documents\Python Scripts\ex_mobility\data\processed\cu_anagrafica.pkl")
    result = get_city_from_coordinates(test_df)
    n_tot = result.shape[0]
    n_right = (result["denominazione_comune_x"] == result["denominazione_comune_y"]).sum()
    perc_success = n_right / n_tot
    log.warning("Test get_city_from_coordinates: {}".format(round(perc_success * 100, 1)))


def test_get_coordinates_from_address():
    path = root_path / PureWindowsPath(r"data_sources/Test/farmacie_italiane.csv")
    test_df = pd.read_csv(path, sep=";", engine='python')
    test_df = test_df.sample(600)
    result = get_coordinates_from_address(test_df, "INDIRIZZO",
                                          city_tag="DESCRIZIONECOMUNE",
                                          province_tag="DESCRIZIONEPROVINCIA",
                                          regione_tag="DESCRIZIONEREGIONE")
    n_tot = result.shape[0]
    n_right = result["test"].sum()
    perc_success = n_right / n_tot
    if perc_success >= 0.7:
        print("OK ", round(perc_success * 100, 1))
    else:
        print("KO ", round(perc_success * 100, 1))


def test_find_coordinates_system():
    test_df = get_df_comuni()
    result = __find_coordinates_system(test_df, "center_y", "center_x")
    log.info("Test1: {}".format(result == "epsg:32632"))

    path = root_path / PureWindowsPath(r"data_sources/Test/farmacie_italiane.csv")
    test_df = pd.read_csv(path, sep=";", engine='python')
    test_df = test_df[test_df["LATITUDINE"] != "-"]
    test_df["LATITUDINE"] = test_df["LATITUDINE"].str.replace(",", ".")
    test_df["LONGITUDINE"] = test_df["LONGITUDINE"].str.replace(",", ".")
    test_df["LATITUDINE"] = test_df["LATITUDINE"].astype(float)
    test_df["LONGITUDINE"] = test_df["LONGITUDINE"].astype(float)
    result = __find_coordinates_system(test_df, "LATITUDINE", "LONGITUDINE")
    log.info("Test2: {}".format(result == "epsg:4326"))


def test_KDEDensity():
    test_df = pd.read_pickle(
        r"C:\Users\A470222\Documents\Python Scripts\ex_mobility\data\Geo/Densita\population_ita_2019-07-01.pkl")
    test_df = test_df[test_df["denominazione_comune"] == "Prato"]

    kde = KDEDensity(test_df, "Lat", "Lon", value_tag="Population")
    prova1 = kde.evaluate_in_point(43.89243338039644, 11.07762361613304)
    prova2 = kde.evaluate_in_point(43.874280169137386, 11.065771494973662)
    prova = "Ciao"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_add_geographic_info()
    #test_get_city_from_coordinates()
    #test_get_coordinates_from_address()
    #test_find_coordinates_system()
    #test_KDEDensity()
