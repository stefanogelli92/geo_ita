from geo_ita.enrich_dataframe import *
from geo_ita.data import create_df_comuni, get_popolazione_df
from geo_ita.definition import *
from pathlib import PureWindowsPath
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def test_add_geographic_info():
    path = root_path / PureWindowsPath(r"data_sources/Test/uniform_name.xlsx")
    test_df = pd.read_excel(path)
    n_tot = test_df.shape[0]

    test_df1 = add_geographic_info(test_df, comuni_tag="Citta", unique_flag=False)
    n_right1 = (test_df1[cfg.TAG_COMUNE].fillna('-').eq(test_df1["comune"].fillna('-'))).sum()
    n_right2 = (test_df1[cfg.TAG_PROVINCIA].fillna('-').eq(test_df1["provincia"].fillna('-'))).sum()
    perc_success = n_right1 / n_tot
    log.info("Test1 get_province_from_city: {}, {}".format(round(perc_success * 100, 1), (n_right2 == n_tot - 1)))
    if perc_success != 1:
        log.warning("\n" + test_df1.to_string())

    test_df2 = add_geographic_info(test_df, comuni_tag="Citta", province_tag="sl", unique_flag=False)
    n_right = ((test_df2[cfg.TAG_COMUNE].fillna('-').eq(test_df2["comune"].fillna('-'))) &
               (test_df2[cfg.TAG_PROVINCIA].fillna('-').eq(test_df2["provincia"].fillna('-')))).sum()
    perc_success = n_right / n_tot
    log.info("Test2 get_province_from_city:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + test_df2.to_string())

    test_df2 = add_geographic_info(test_df, comuni_tag="Citta", province_tag="provincia", unique_flag=False)
    n_right = ((test_df2[cfg.TAG_COMUNE].fillna('-').eq(test_df2["comune"].fillna('-'))) &
               (test_df2[cfg.TAG_PROVINCIA].fillna('-').eq(test_df2["provincia"].fillna('-')))).sum()
    perc_success = n_right / n_tot
    log.info("Test3 get_province_from_city:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + test_df2.to_string())

    test_df3 = add_geographic_info(test_df, comuni_tag="Citta", regioni_tag="regione", unique_flag=False)
    n_right = ((test_df3[cfg.TAG_COMUNE].fillna('-').eq(test_df3["comune"].fillna('-'))) &
               (test_df3[cfg.TAG_PROVINCIA].fillna('-').eq(test_df3["provincia"].fillna('-')))).sum()
    perc_success = n_right / n_tot
    log.info("Test4 get_province_from_city:{}".format(round(perc_success * 100, 1)))
    if perc_success < 1:
        log.warning("\n" + test_df3.to_string())


def test_get_city_from_coordinates():
    test_df = create_df_comuni()
    test_df.rename(columns={"center_x": "lat",
                            "center_y": "lon"}, inplace=True)
    result = get_city_from_coordinates(test_df)
    n_tot = result.shape[0]
    n_right = (result["denominazione_comune_x"] == result["denominazione_comune_y"]).sum()
    perc_success = n_right / n_tot
    log.warning("Test get_city_from_coordinates: {}".format(round(perc_success * 100, 1)))




def test_get_coordinates_from_address():
    path = root_path / PureWindowsPath(r"data_sources/Test/farmacie_italiane.csv")
    test_df = pd.read_csv(path, sep=";", engine='python')
    test_df = test_df.sample(100)
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_add_geographic_info()
    test_get_city_from_coordinates()
    test_get_coordinates_from_address()
