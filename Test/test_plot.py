from geo_ita.plot import *
from geo_ita.data import create_df_comuni
import geo_ita.config as cfg


def test_plot_region_distribution():
    test_df = create_df_comuni()
    # Aggiungo una finta categoria per i plot non quantitativi
    test_df["prima_lettera"] = test_df[cfg.TAG_REGIONE].str[0]

    plot_region_distribution(test_df[[cfg.TAG_REGIONE, "prima_lettera"]].drop_duplicates(), cfg.TAG_REGIONE, "prima_lettera")
    print("ok1")
    plot_province_distribution(test_df, cfg.TAG_PROVINCIA, "popolazione", print_labels=True, print_perc=False)
    print("ok2")
    plot_comuni_distribution(test_df, cfg.TAG_COMUNE, "popolazione", filter_regioni=["lazio ", "campania"])
    print("ok3")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("Popolazione")
    plot_region_distribution(test_df, "denominazione_regione", "popolazione", ax=ax[0], print_labels=True,
                             print_perc=False)
    plot_region_distribution(test_df, "denominazione_regione", "superficie_km2", ax=ax[1], print_labels=True,
                             print_perc=False)
    ax[1].set_title("Superficie")
    plt.show()
    print("ok4")


if __name__ == '__main__':
    test_plot_region_distribution()

