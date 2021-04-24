from geo_ita.plot import *
from geo_ita.data import create_df_comuni
import geo_ita.config as cfg


def test_1():

    test_df = create_df_comuni()
    # Aggiungo una finta categoria per i plot non quantitativi
    test_df["prima_lettera"] = test_df[cfg.TAG_REGIONE].str[0]

   # plot_choropleth_map_regionale(test_df[[cfg.TAG_REGIONE, "prima_lettera"]].drop_duplicates(), cfg.TAG_REGIONE, "prima_lettera")
    print("ok1")
    #plot_choropleth_map_provinciale(test_df, cfg.TAG_PROVINCIA, "popolazione", print_labels=True, print_perc=False)
    print("ok2")
    plot_choropleth_map_comunale(test_df, cfg.TAG_COMUNE, "popolazione", filter_regioni=["lazio ", "campania"])
    print("ok3")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("Popolazione")
    plot_choropleth_map_regionale(test_df, "denominazione_regione", "popolazione", ax=ax[0], print_labels=True,
                             print_perc=False)
    plot_choropleth_map_regionale(test_df, "denominazione_regione", "superficie_km2", ax=ax[1], print_labels=True,
                             print_perc=False)
    ax[1].set_title("Superficie")
    plt.show()
    print("ok4")


def test_2():
    test_df = create_df_comuni().drop(["geometry"], axis=1)
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
    plot_choropleth_map_provinciale_interactive(test_df.groupby("denominazione_provincia")["popolazione"].sum().reset_index(),
                                                "denominazione_provincia",
                                                {"popolazione": "Popolazione"})
    plot_choropleth_map_comunale_interactive(test_df,
                                              "denominazione_comune",
                                              {"popolazione": "Popolazione",
                                               "superficie_km2": "Superficie",
                                               "prima_lettera": "Prima Lettera"},
                                               filter_regioni=["Toscana"])

if __name__ == '__main__':
    test_1()
    #test_2()

