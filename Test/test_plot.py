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

    def test_plot_choropleth_map(self):
        test_df = get_df_regioni()

        plot_choropleth_map_regionale(test_df, cfg.TAG_REGIONE,
                                      "popolazione",
                                      title="Popolazione Regionale", save_path="usage_choropleth_regionale.png",
                                      dpi=50)
        print("ok1")

        test_df = get_df_comuni()
        plot_choropleth_map_comunale_interactive(test_df, cfg.TAG_COMUNE, {"popolazione": "Popolazione",
                                                                                 "superficie_km2": "Superficie"},
                                                    title="Semplificato",
                                                    save_path="comunale_semplificato.html")
        print("ok2")
        plot_choropleth_map_comunale(test_df, cfg.TAG_COMUNE, "popolazione", filter_regione=["lazio ", "campania"])
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

    def test_point_map(self):
        test_df = get_df_province()
        plot_point_map_interactive(test_df,
                                   longitude_columns="center_x",
                                   latitude_columns="center_y",
                                   filter_regione="Toscana", show_flag=False,
                                   save_in_path="usage_point_map_1.html")
        plot_point_map(test_df, latitude_columns='center_y', longitude_columns='center_x',
                        size=12, title="Province", save_in_path="usage_point_map_comuni.png",
                        color_tag="popolazione")
        test_df = get_df_comuni()
        plot_point_map(test_df, latitude_columns='center_y', longitude_columns='center_x',
                       filter_regione="Toscana", color_tag="denominazione_provincia",
                       size=8, title="Comuni", save_in_path="usage_point_map_comuni2.png")
        plot_point_map(test_df, latitude_columns='center_y', longitude_columns='center_x', color_tag="popolazione", filter_provincia="Prato")
        plot_point_map(test_df, latitude_columns='center_y', longitude_columns='center_x', color_tag="denominazione_regione", legend_font=5)
        plot_point_map_interactive(test_df, latitude_columns='center_y', longitude_columns='center_x',
                                   color_tag="denominazione_regione", save_in_path="usage_point_map_2.html")

    def test_density(self):
        #df = get_high_resolution_population_density_df()
        #plot_kernel_density_estimation(df, n_grid_x=500, n_grid_y=500)

        test_df = get_df_comuni()
        plot_kernel_density_estimation(test_df, latitude_columns='center_y', longitude_columns='center_x',
                                       n_grid_x=500, n_grid_y=500,
                                       save_in_path="usage_kernel_density_simple.png")
        plot_kernel_density_estimation(test_df, value_tag="popolazione", latitude_columns='center_y', longitude_columns='center_x',
                                       n_grid_x=500, n_grid_y=500,
                                       save_in_path="usage_kernel_density_variable.png")
        plot_kernel_density_estimation_interactive(test_df, value_tag="popolazione", latitude_columns='center_y',
                                       longitude_columns='center_x',
                                       n_grid_x=500, n_grid_y=500,
                                       filter_regione="Lazio",
                                                   save_in_path="usage_kernel_density_estimation_1.html")



