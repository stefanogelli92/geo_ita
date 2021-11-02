from geo_ita.data import *


def test_run():
    prova_1 = get_df_comuni()
    print(get_df_comuni()[["denominazione_comune", "center_x", "center_y", "popolazione"]])
    prova_2 = get_df_province()
    prova_3 = get_df_regioni()
    prova_4 = get_list_comuni()
    prova_5 = get_list_province()
    prova_6 = get_list_regioni()

    prova = "Debug"


if __name__ == '__main__':
    test_run()
