import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import geopandas as gpd
from geo_ita.data import create_df_comuni, create_df_province, create_df_regioni, get_anagrafica_df
from geo_ita.enrich_dataframe import __clean_denom_text, __uniform_names
import geo_ita.config as cfg
from pandas.api.types import is_numeric_dtype

CODE_CODICE_ISTAT = 0
CODE_SIGLA = 1
CODE_DENOMINAZIONE = 2
LEVEL_COMUNE = 0
LEVEL_PROVINCIA = 1
LEVEL_REGIONE = 2


def _human_format(num):
    num = float('{:.2g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def _linear_colormap(color_name1="white", color_name2=None, minval=0, maxval=1):
    if color_name2 is None:
        color_name2 = "blue"
    cmap = _truncate_colormap(LinearSegmentedColormap.from_list("", [color_name1, color_name2]), minval=minval,
                              maxval=maxval)
    return cmap


def _plot_distribution_on_shape(df, color, ax, show_colorbar, numeric_values, value_tag, linewidth=0.8):
    if numeric_values:
        vmin, vmax = df["count"].min(), df["count"].max()
        cmap = _linear_colormap(color_name2=color)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
    else:
        cmap = get_cmap('tab10')
        color_map = df["count"].value_counts(dropna=False).reset_index()
        color_map["color"] = None
        for i in range(color_map.shape[0]):
            color_map.iat[i, 2] = cmap(i % 10)
        df["color"] = df["count"].map(color_map.set_index("index")["color"])
        legend_elements = [Patch(facecolor=row.color, label=row["index"]) for index, row in color_map.iterrows()]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 10))
        if show_colorbar & numeric_values:
            cbar = fig.colorbar(sm)
            cbar.ax.tick_params(labelsize=10)
    ax.axis('off')
    if numeric_values:
        df.plot('count', cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth, edgecolor='0.8', ax=ax)
    else:
        df.plot('count', linewidth=linewidth, edgecolor='0.8', ax=ax, color=df["color"].values)
        ax.legend(title=value_tag, handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

    return fig, ax


def _add_labels_on_plot(df, ax, print_perc, numeric_values):
    if numeric_values:
        total = df["count"].sum()
        for idx, row in df.iterrows():
            threshold = 0
            if print_perc:
                if row['count'] > threshold:
                    ax.annotate(text=round(row['count'] / total * 100, 1).astype(str) + "%",
                                xy=(row['center_x'], row['center_y']),
                                horizontalalignment='center', fontsize='large', color='black', wrap=True)
            else:
                if row['count'] > threshold:
                    ax.annotate(text=_human_format(row['count']), xy=(row['center_x'], row['center_y']),
                                horizontalalignment='center', fontsize='large', color='black', wrap=True)
    else:
        for idx, row in df.iterrows():
            ax.annotate(text=row['count'],
                        xy=(row['center_x'], row['center_y']),
                        horizontalalignment='center', fontsize='large', color='black', wrap=True)


def _code_or_desc(list_values):
    n_tot = len(list_values)
    if (sum([isinstance(item, int) or item.isdigit() for item in list_values]) / n_tot) > 0.8:
        result = CODE_CODICE_ISTAT
    elif (sum([isinstance(item, str) and item.isalpha() and len(item) == 2 for item in list_values]) / n_tot) > 0.8:
        result = CODE_SIGLA
    else:
        result = CODE_DENOMINAZIONE
    return result


def _get_tag_anag(code, level):
    if level == LEVEL_COMUNE:
        if code == CODE_CODICE_ISTAT:
            result = cfg.TAG_CODICE_COMUNE
        else:
            result = cfg.TAG_COMUNE
    elif level == LEVEL_PROVINCIA:
        if code == CODE_CODICE_ISTAT:
            result = cfg.TAG_CODICE_PROVINCIA
        elif code == CODE_SIGLA:
            result = cfg.TAG_SIGLA
        else:
            result = cfg.TAG_PROVINCIA
    elif level == LEVEL_REGIONE:
        if code == CODE_CODICE_ISTAT:
            result = cfg.TAG_CODICE_REGIONE
        else:
            result = cfg.TAG_REGIONE
    else:
        raise Exception("Level UNKNOWN")
    return result


def _get_shape_from_level(level):
    if level == LEVEL_COMUNE:
        shape = create_df_comuni()
    elif level == LEVEL_PROVINCIA:
        shape = create_df_province()
    elif level == LEVEL_REGIONE:
        shape = create_df_regioni()
    else:
        raise Exception("Level UNKNOWN")
    return shape


def _plot_distribution(df0,
                       geo_tag1,
                       value_tag,
                       level,
                       color,
                       ax,
                       show_colorbar,
                       print_labels,
                       print_perc,
                       filter_list=None,
                       level2=None):
    df = df0.copy()
    df = df[df[geo_tag1].notnull()]

    code = _code_or_desc(list(df[geo_tag1].unique()))

    shape = _get_shape_from_level(level)

    geo_tag2 = _get_tag_anag(code, level)

    anagr_df = get_anagrafica_df()

    if filter_list is not None:
        code_filter = _code_or_desc(filter_list)
        tag_2 = _get_tag_anag(code_filter, level2)
        if code_filter == CODE_CODICE_ISTAT:
            filter_list = [int(x) for x in filter_list]
        elif code_filter == CODE_SIGLA:
            filter_list = [x.upper() for x in filter_list]
        else:
            filter_list = pd.DataFrame(columns=[tag_2], data=filter_list)
            filter_list, anagr_df = __uniform_names(filter_list, shape, tag_2, tag_2, tag_2)
            filter_list = filter_list[tag_2].unique()
        anagr_df = anagr_df[anagr_df[tag_2].isin(filter_list)]

    if code == CODE_DENOMINAZIONE:
        df, shape = __uniform_names(df, shape, geo_tag1, geo_tag2, geo_tag1)

    if filter_list is not None:
        anagr_df[geo_tag2] = __clean_denom_text(anagr_df[geo_tag2])
        shape = shape[shape[geo_tag2].isin(anagr_df[geo_tag2].unique())]

    numeric_values = is_numeric_dtype(df[value_tag])

    if numeric_values:
        df = df.groupby(geo_tag1)[value_tag].sum()
    else:
        # Test unique values
        if df[geo_tag1].nunique() == df.shape[0]:
            df = df.set_index(geo_tag1)[value_tag]
        else:
            raise Exception("When you want to plot a cathegorical values you need to group by your geographical area.")

    shape["count"] = shape[geo_tag2].map(df)
    if numeric_values:
        shape["count"].fillna(0, inplace=True)
    shape = gpd.GeoDataFrame(shape, geometry="geometry")
    fig, ax = _plot_distribution_on_shape(shape, color, ax, show_colorbar, numeric_values, value_tag)

    if print_labels:
        _add_labels_on_plot(shape, ax, print_perc, numeric_values)

    if fig is not None:
        plt.show()

    return fig


def plot_region_distribution(df,
                             region_tag,
                             value_tag,
                             ax=None,
                             color="b",
                             show_colorbar=True,
                             print_labels=True,
                             filter_regioni=None,
                             print_perc=True):
    _plot_distribution(df,
                       region_tag,
                       value_tag,
                       LEVEL_REGIONE,
                       color,
                       ax,
                       show_colorbar,
                       print_labels,
                       print_perc,
                       filter_list=filter_regioni,
                       level2=LEVEL_REGIONE)


def plot_province_distribution(df,
                               province_tag,
                               value_tag,
                               ax=None,
                               color="b",
                               show_colorbar=True,
                               print_labels=False,
                               filter_regioni=None,
                               filter_province=None,
                               print_perc=True):
    if filter_regioni:
        level_filter = LEVEL_REGIONE
        filter_list = filter_regioni
    elif filter_province:
        level_filter = LEVEL_PROVINCIA
        filter_list = filter_province
    else:
        level_filter = None
        filter_list = None
    _plot_distribution(df,
                       province_tag,
                       value_tag,
                       LEVEL_PROVINCIA,
                       color,
                       ax,
                       show_colorbar,
                       print_labels,
                       print_perc,
                       filter_list=filter_list,
                       level2=level_filter)


def plot_comuni_distribution(df,
                             comuni_tag,
                             value_tag,
                             ax=None,
                             color="b",
                             show_colorbar=True,
                             print_labels=False,
                             filter_regioni=None,
                             filter_province=None,
                             filter_comuni=None,
                             print_perc=True):
    if filter_regioni:
        level_filter = LEVEL_REGIONE
        filter_list = filter_regioni
    elif filter_province:
        level_filter = LEVEL_PROVINCIA
        filter_list = filter_province
    elif filter_comuni:
        level_filter = LEVEL_COMUNE
        filter_list = filter_comuni
    else:
        level_filter = None
        filter_list = None
    _plot_distribution(df,
                       comuni_tag,
                       value_tag,
                       LEVEL_COMUNE,
                       color,
                       ax,
                       show_colorbar,
                       print_labels,
                       print_perc,
                       filter_list=filter_list,
                       level2=level_filter)