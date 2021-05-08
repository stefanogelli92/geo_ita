from geo_ita.src._data import *


def test_run():
    get_anagrafica_df()
    get_popolazione_df()
    get_comuni_shape_df()
    get_province_shape_df()
    get_regioni_shape_df()
    get_dimensioni_df()
    create_df_comuni()
    create_df_province()
    create_df_regioni()


if __name__ == '__main__':
    test_run()
