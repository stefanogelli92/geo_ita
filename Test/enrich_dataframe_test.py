from geo_ita.enrich_dataframe import *
from geo_ita.data import create_df_comuni, get_popolazione_df
from geo_ita.definition import *
from pathlib import PureWindowsPath


def test_get_city_from_coordinates():
    test_df = create_df_comuni()
    test_df.rename(columns={"center_x": "lat",
                            "center_y": "lon"}, inplace=True)
    result = get_city_from_coordinates(test_df)
    n_tot = result.shape[0]
    n_right = (result["denominazione_comune_x"] == result["denominazione_comune_y"]).sum()
    perc_success = n_right / n_tot
    if perc_success >= 0.8:
        print("OK ", round(perc_success * 100, 1))
    else:
        print("KO ", round(perc_success * 100, 1))


def test_get_province_from_city():
    test_df = get_popolazione_df()
    result = get_province_from_city(test_df, "denominazione_comune", unique_flag=True)
    n_tot = result.shape[0]
    n_right = result["denominazione_provincia"].notnull().sum()
    perc_success = n_right / n_tot
    if perc_success >= 0.8:
        print("OK ", round(perc_success * 100, 1))
    else:
        print("KO ", round(perc_success * 100, 1))


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
    #test_get_city_from_coordinates()
    #test_get_province_from_city()
    test_get_coordinates_from_address()
