from geo_ita.src._plot import *
from geo_ita.src._data import get_df_comuni, _get_shape_italia
import geo_ita.src.config as cfg
from geo_ita.src.definition import *
from pathlib import PureWindowsPath


def test_plot_choropleth_map():
    test_df = get_df_comuni()
    # Aggiungo una finta categoria per i plot non quantitativi
    test_df["prima_lettera"] = test_df[cfg.TAG_REGIONE].str[0]

    plot_choropleth_map_regionale(test_df[[cfg.TAG_REGIONE, "prima_lettera"]].drop_duplicates(), cfg.TAG_REGIONE,
                                  "prima_lettera")
    print("ok1")
    plot_choropleth_map_provinciale(test_df, cfg.TAG_PROVINCIA, "popolazione",
                                    print_labels=True, print_perc=False, labels_size=5)
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
    test_df = get_df_comuni()
    test_df.rename(columns={"center_x": "lon",
                            "center_y": "lat"}, inplace=True)
    plot_point_map(test_df, size=1)
    plot_point_map(test_df, color_tag="popolazione", provincia="Prato")
    plot_point_map(test_df, color_tag="denominazione_regione", legend_font=5)
    plot_point_map_interactive(test_df, color_tag="denominazione_regione")
    plot_point_map_interactive(test_df, provincia="Prato")



def test_4():

    # path = root_path / PureWindowsPath(r"data_sources/Test/farmacie_italiane.csv")
    # test_df = pd.read_csv(path, sep=";", engine='python')
    # test_df = test_df[test_df["LATITUDINE"] != "-"]
    # test_df["LATITUDINE"] = test_df["LATITUDINE"].str.replace(",", ".")
    # test_df["LONGITUDINE"] = test_df["LONGITUDINE"].str.replace(",", ".")
    # plot_kernel_density_estimation_interactive(test_df, n_grid_x=50, n_grid_y=50, comune="Prato")

    test_df = pd.read_pickle(r"C:\Users\A470222\Documents\Python Scripts\ex_mobility\data\Geo/Densita\population_ita_2019-07-01.pkl")
    #plot_kernel_density_estimation_interactive(test_df)
    test_df = test_df[test_df["denominazione_comune"] == "Prato"]
    pop_total = test_df["Population"].sum()
    #plot_point_map_interactive(test_df, comune="Prato", info_dict={"Population": "Popolazione"})
    plot_kernel_density_estimation_interactive(test_df, value_tag="Population", provincia="Roma")
    #plot_kernel_density_estimation(test_df)


    #plot_kernel_density_estimation_interactive(test_df, value_tag="Population", regione="Toscana")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_plot_choropleth_map()
    #test_2()
    #test_point_map()
    #test_4()
