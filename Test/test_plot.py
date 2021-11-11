from geo_ita.src._plot import *
from geo_ita.src._plot import _get_margins
from geo_ita.src._data import get_df_comuni, _get_shape_italia, get_high_resolution_population_density_df
import geo_ita.src.config as cfg
from geo_ita.src.definition import *
from pathlib import PureWindowsPath

import unittest


class TestPlot(unittest.TestCase):

    # plot_choropleth_map

    def test_get_marging(self):
        with self.assertRaises(Exception):
            _get_margins(1)
        result, _ = _get_margins()
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result), 4)
        result1, _ = _get_margins(filter_comune="PRato")
        result2, _ = _get_margins(filter_comune=["PrAto"])
        self.assertListEqual(result1, result2)
        result3, _ = _get_margins(filter_comune=["Prato", "firenze"])
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result), 4)
        self.assertTrue(result2 != result3)

    def test_plot_choropleth_map_input(self):
        pass


def test_plot_choropleth_map():
    test_df = get_df_regioni()

    plot_choropleth_map_regionale(test_df, cfg.TAG_REGIONE,
                                  "popolazione",
                                  title="Popolazione Regionale", save_path="usage_choropleth_regionale.png",
                                  dpi=50)
    print("ok1")
    test_df = get_df_province()
    plot_choropleth_map_provinciale_interactive(test_df, cfg.TAG_PROVINCIA, {"popolazione": "Popolazione",
                                                                             "superficie_km2": "Superficie"},
                                                filter_regione="Toscana",
                                                title="Toscana")
    print("ok2")
    plot_choropleth_map_comunale(test_df, cfg.TAG_COMUNE, "popolazione", filter_regioni=["lazio ", "campania"])
    print("ok3")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("Popolazione")
    plot_choropleth_map_regionale(test_df, "denominazione_regione", "popolazione", ax=ax[0], print_labels=True,
                                  print_perc=False)
    plot_choropleth_map_regionale(test_df, "denominazione_regione", "superficie_km2", ax=ax[1], print_labels=True,
                                  print_perc=False, labels_size=15)
    ax[1].set_title("Superficie")
    plt.show()
    print("ok4")


def test_2():
    test_df = get_df_comuni().drop(["geometry"], axis=1)
    # Aggiungo una finta categoria per i plot non quantitativi
    test_df["prima_lettera"] = test_df[cfg.TAG_REGIONE].str[0]
    plot_choropleth_map_comunale_interactive(test_df.sample(100),
                                             "denominazione_comune",
                                             {"popolazione": "Popolazione",
                                              "superficie_km2": "Superficie",
                                              "prima_lettera": "Prima Lettera"})

    plot_choropleth_map_regionale_interactive(test_df[[cfg.TAG_REGIONE, "prima_lettera"]].drop_duplicates(),
                                              "denominazione_regione",
                                              {"prima_lettera": "Prima Lettera"})
    plot_choropleth_map_provinciale_interactive(
        test_df.groupby("denominazione_provincia")["popolazione"].sum().reset_index(),
        "denominazione_provincia",
        {"popolazione": "Popolazione"})
    plot_choropleth_map_comunale_interactive(test_df,
                                             "denominazione_comune",
                                             {"popolazione": "Popolazione",
                                              "superficie_km2": "Superficie",
                                              "prima_lettera": "Prima Lettera"},
                                             filter_regioni=["Toscana"],
                                             title="Toscana")


def test_point_map():
    # margins0 = _get_margins()
    # margins1 = _get_margins(comune="PRato")
    # margins3 = _get_margins(provincia="Firenze")
    # margins4 = _get_margins(regione="Toscana")
    # margins5 = _get_margins(regione="Tascana")
    test_df = get_df_province()
    plot_point_map_interactive(test_df,
                               longitude_columns="center_x",
                               latitude_columns="center_y",
                               filter_regione="Toscana")
    plot_point_map(test_df, latitude_columns='center_y', longitude_columns='center_x',
                   size=12, title="Province", save_in_path="usage_point_map_comuni.png",
                   color_tag="popolazione")
    test_df = get_df_comuni()
    plot_point_map(test_df, latitude_columns='center_y', longitude_columns='center_x',
                   regione="Toscana", color_tag="denominazione_provincia",
                   size=8, title="Comuni", save_in_path="usage_point_map_comuni2.png")
    plot_point_map(test_df, color_tag="popolazione", provincia="Prato")
    plot_point_map(test_df, color_tag="denominazione_regione", legend_font=5)
    plot_point_map_interactive(test_df, color_tag="denominazione_regione")


def test_density():
    df = get_high_resolution_population_density_df()
    plot_kernel_density_estimation(df, n_grid_x=500, n_grid_y=500)



    # n = 1000
    # df = pd.DataFrame(index=range(n))
    #
    # df["lat"] = np.random.normal((42 + 41.8)/2, (42 - 41.8)/10,  size=(n, 1))
    # df["lon"] = np.random.normal((12.6 + 12.37)/2, (12.6 - 12.37)/10, size=(n, 1))
    # df["popolazione"] = np.random.uniform(0, 100, size=(n, 1))
    # test_df = get_df_comuni()
    # plot_kernel_density_estimation(test_df, latitude_columns='center_y', longitude_columns='center_x',
    #                                n_grid_x=500, n_grid_y=500,
    #                                save_in_path="usage_kernel_density_simple.png")
    # plot_kernel_density_estimation(test_df, value_tag="popolazione", latitude_columns='center_y', longitude_columns='center_x',
    #                                n_grid_x=500, n_grid_y=500,
    #                                save_in_path="usage_kernel_density_variable.png")
    # plot_kernel_density_estimation_interactive(test_df, value_tag="popolazione", latitude_columns='center_y',
    #                                longitude_columns='center_x',
    #                                n_grid_x=500, n_grid_y=500,
    #                                filter_regione="Lazio")
    #
    #
    #
    # # path = root_path / PureWindowsPath(r"data_sources/Test/farmacie_italiane.csv")
    # # test_df = pd.read_csv(path, sep=";", engine='python')
    # # test_df = test_df[test_df["LATITUDINE"] != "-"]
    # # test_df["LATITUDINE"] = test_df["LATITUDINE"].str.replace(",", ".")
    # # test_df["LONGITUDINE"] = test_df["LONGITUDINE"].str.replace(",", ".")
    # # plot_kernel_density_estimation_interactive(test_df, n_grid_x=50, n_grid_y=50, comune="Prato")
    #
    # test_df = pd.read_pickle(r"C:\Users\A470222\Documents\Python Scripts\ex_mobility\data\Geo/Densita\population_ita_2019-07-01.pkl")
    # #plot_kernel_density_estimation_interactive(test_df)
    # test_df = test_df[test_df["denominazione_comune"] == "Prato"]
    # pop_total = test_df["Population"].sum()
    # #plot_point_map_interactive(test_df, comune="Prato", info_dict={"Population": "Popolazione"})
    # plot_kernel_density_estimation_interactive(test_df, value_tag="Population", provincia="Roma")
    # #plot_kernel_density_estimation(test_df)


    #plot_kernel_density_estimation_interactive(test_df, value_tag="Population", regione="Toscana")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #test_plot_choropleth_map()
    #test_point_map()
    test_density()
    unittest.main()
    #test_2()


