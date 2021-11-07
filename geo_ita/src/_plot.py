import os
import logging

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.ndimage.filters
from pandas.api.types import is_numeric_dtype, is_string_dtype
from bokeh.palettes import Blues9, Greens9, Reds9, Greys9, Purples9, Oranges9, Category10, \
    Category20, RdYlGn11
from bokeh.plotting import save, figure
from bokeh.layouts import column, row
from bokeh.io import output_file, show
from bokeh.models.mappers import LinearColorMapper
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.models import (ColumnDataSource, Circle,
                          WheelZoomTool, HoverTool, DataTable, TableColumn, Select,
                          CustomJS, GeoJSONDataSource, ColorBar,
                          CategoricalColorMapper, NumberFormatter, NumeralTickFormatter)

import geo_ita.src.config as cfg
from geo_ita.src._data import get_df_comuni, get_df_province, get_df_regioni, _get_shape_italia
from geo_ita.src._enrich_dataframe import _clean_denom_text_value, _clean_denom_text, _get_tag_anag, _code_or_desc, \
    AddGeographicalInfo, __create_geo_dataframe, __find_coord_columns, __find_coordinates_system
from pyproj import Proj, transform

HEADER_BOKEH = {cfg.LEVEL_COMUNE: 'Comune',
                cfg.LEVEL_PROVINCIA: 'Provincia',
                cfg.LEVEL_REGIONE: 'Regione'}

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def _human_format(num):
    # Show float number in more readble format
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
    # Create a 2 color linear map
    if color_name2 is None:
        color_name2 = "blue"
    cmap = _truncate_colormap(LinearSegmentedColormap.from_list("", [color_name1, color_name2]), minval=minval,
                              maxval=maxval)
    return cmap


def _plot_choropleth_map(df, color, ax, title, show_colorbar, vmin, vmax, numeric_values, value_tag, prefix, suffix, line_width=0.8, shape_list=[]):
    if numeric_values:
        if vmin is None:
            vmin = df["count"].min()
        if vmax is None:
            vmax = df["count"].max()
        cmap = _linear_colormap(color_name2=color)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
    else:
        color_map = df["count"].value_counts(dropna=False).reset_index()
        if color_map.shape[0] <= 10:
            cmap = get_cmap('tab10')
        else:
            cmap = get_cmap('tab20')
        color_map["color"] = None
        for i in range(color_map.shape[0]):
            color_map.iat[i, 2] = cmap(i % 10)
        df["color"] = df["count"].map(color_map.set_index("index")["color"])
        legend_elements = [Patch(facecolor=row.color, label=row["index"]) for index, row in color_map.iterrows()]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 10))
        if title:
            ax.set_title(title)

    ax.axis('off')
    if numeric_values:
        df.plot('count', cmap=cmap, vmin=vmin, vmax=vmax, linewidth=line_width, edgecolor='0.8', ax=ax)
        if show_colorbar & numeric_values:
            fmt = lambda x, pos: str(prefix) + _human_format(x) + str(suffix)
            cbar = fig.colorbar(sm, format=FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=10)
    else:
        df.plot('count', linewidth=line_width, edgecolor='0.8', ax=ax, color=df["color"].values)
        ax.legend(title=value_tag, handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

    edgecolor = "0.6"
    for shape, lw in shape_list:
        shape.plot(facecolor="none", linewidth=lw, edgecolor=edgecolor, ax=ax)
        edgecolor = "0.4"

    return fig, ax


def _add_labels_on_plot(df, ax, print_perc, numeric_values, prefix, suffix, labels_size):
    if labels_size is None:
        labels_size = 'large'
    if numeric_values:
        total = df["count"].sum()
        max_value = df["count"].max()
        min_value = df["count"].min()
        df["tx_color"] = np.where(df["count"]>((max_value+min_value)/2), "white", "black")
        for idx, row in df.iterrows():
            threshold = 0
            if print_perc:
                if row['count'] > threshold:
                    ax.annotate(text=round(row['count'] / total * 100, 1).astype(str) + "%",
                                xy=(row['center_x'], row['center_y']),
                                ha='center', va="center",
                                fontsize=labels_size, color=row["tx_color"], wrap=True)

            else:
                if row['count'] > threshold:
                    ax.annotate(text=str(prefix) + _human_format(row['count']) + str(suffix), xy=(row['center_x'], row['center_y']),
                                ha='center', va="center",
                                color=row["tx_color"], wrap=True, fontsize=labels_size)
    else:
        for idx, row in df.iterrows():
            ax.annotate(text=row['count'],
                        xy=(row['center_x'], row['center_y']),
                        ha='center', va="center",
                        color='black', wrap=True, fontsize=labels_size)


def _get_shape_from_level(level):
    if level == cfg.LEVEL_COMUNE:
        shape = get_df_comuni()
    elif level == cfg.LEVEL_PROVINCIA:
        shape = get_df_province()
    elif level == cfg.LEVEL_REGIONE:
        shape = get_df_regioni()
    else:
        raise Exception("Level UNKNOWN")
    return shape


def _check_filter(filter_list):
    if filter_list is None:
        result = None
    elif isinstance(filter_list, str):
        result = [filter_list]
    elif isinstance(filter_list, list):
        result = filter_list
    else:
        raise Exception("Filter not recognized. You can use a string or a list of string as filter.")
    return result


def _create_choropleth_map(df0,
                           geo_tag_input,
                           value_tag,
                           level,
                           color,
                           ax,
                           title,
                           show_colorbar,
                           vmin,
                           vmax,
                           print_labels,
                           prefix,
                           suffix,
                           print_perc,
                           filter_list=None,
                           level2=None,
                           labels_size=None,
                           save_path=None,
                           dpi=100):
    filter_list = _check_filter(filter_list)
    # Todo add unità di misura labels / clorobar
    # Todo Cambio nome legenda
    # Todo Set title
    # Todo Plot backgroud regions grey
    # Todo auto check if center scale and use 3 color map
    df = df0.copy()
    df = df[df[geo_tag_input].notnull()]

    code = _code_or_desc(list(df[geo_tag_input].unique()))

    shape = _get_shape_from_level(level)

    geo_tag_anag = _get_tag_anag(code, level)

    geoInf = AddGeographicalInfo(df)
    if level == cfg.LEVEL_COMUNE:
        geoInf.set_comuni_tag(geo_tag_input)
    elif level == cfg.LEVEL_PROVINCIA:
        geoInf.set_province_tag(geo_tag_input)
    elif level == cfg.LEVEL_REGIONE:
        geoInf.set_regioni_tag(geo_tag_input)

    geoInf.run_simple_match()
    if level == cfg.LEVEL_COMUNE:
        geoInf.run_find_frazioni()
        geoInf.run_similarity_match(unique_flag=False)
        geoInf.accept_similarity_result()
    df = geoInf.get_result()
    del geoInf

    if filter_list is not None:
        code_filter = _code_or_desc(filter_list)
        tag_2 = _get_tag_anag(code_filter, level2)
        if code_filter == cfg.CODE_CODICE_ISTAT:
            filter_list = [int(x) for x in filter_list]
        elif code_filter == cfg.CODE_SIGLA:
            filter_list = [x.lower() for x in filter_list]
        else:
            filter_list = [_clean_denom_text_value(x) for x in filter_list]

        shape = shape[_clean_denom_text(shape[tag_2]).isin(filter_list)]

    numeric_values = is_numeric_dtype(df[value_tag])

    if numeric_values:
        df = df.groupby(geo_tag_anag)[value_tag].sum()
    else:
        # Test unique values
        if df[geo_tag_anag].nunique() == df.shape[0]:
            df = df.set_index(geo_tag_anag)[value_tag]
        else:
            raise Exception("When you want to plot a cathegorical values you need to group by your geographical area.")
    log.debug(df.head(5), geo_tag_anag)
    shape["count"] = shape[geo_tag_anag].map(df)

    if numeric_values:
        shape["count"].fillna(0, inplace=True)
    shape = gpd.GeoDataFrame(shape, geometry="geometry")

    shape_list = []
    if level == cfg.LEVEL_COMUNE:
        line_width = 0.2
        shape_province = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        shape_province = shape_province[shape_province[cfg.TAG_PROVINCIA].isin(shape[cfg.TAG_PROVINCIA].unique())]
        shape_province = gpd.GeoDataFrame(shape_province, geometry="geometry")
        shape_list.append((shape_province, 0.4))
        shape_regioni = _get_shape_from_level(cfg.LEVEL_REGIONE)
        shape_regioni = shape_regioni[shape_regioni[cfg.TAG_REGIONE].isin(shape[cfg.TAG_REGIONE].unique())]
        shape_regioni = gpd.GeoDataFrame(shape_regioni, geometry="geometry")
        shape_list.append((shape_regioni, 0.8))
    elif level == cfg.LEVEL_PROVINCIA:
        line_width = 0.4
        shape_regioni = _get_shape_from_level(cfg.LEVEL_REGIONE)
        shape_regioni = shape_regioni[shape_regioni[cfg.TAG_REGIONE].isin(shape[cfg.TAG_REGIONE].unique())]
        shape_regioni = gpd.GeoDataFrame(shape_regioni, geometry="geometry")
        shape_list.append((shape_regioni, 0.8))
    elif level == cfg.LEVEL_REGIONE:
        line_width = 0.8
    else:
        line_width = 0.2
    log.debug(shape.head(5))
    fig, ax = _plot_choropleth_map(shape,
                                   color,
                                   ax,
                                   title,
                                   show_colorbar,
                                   vmin,
                                   vmax,
                                   numeric_values,
                                   value_tag,
                                   prefix,
                                   suffix,
                                   line_width=line_width,
                                   shape_list=shape_list)

    if print_labels:
        _add_labels_on_plot(shape, ax, print_perc, numeric_values, prefix, suffix, labels_size=labels_size)

    if fig is not None:
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        else:
            plt.show()

    return fig


def plot_choropleth_map_regionale(df,
                                  region_tag,
                                  value_tag,
                                  ax=None,
                                  title=None,
                                  color="b",
                                  show_colorbar=True,
                                  vmin=None,
                                  vmax=None,
                                  print_labels=True,
                                  prefix="",
                                  suffix="",
                                  filter_regione=None,
                                  print_perc=False,
                                  labels_size=None,
                                  save_path=None,
                                  dpi=100):
    _create_choropleth_map(df,
                           region_tag,
                           value_tag,
                           cfg.LEVEL_REGIONE,
                           color,
                           ax,
                           title,
                           show_colorbar,
                           vmin,
                           vmax,
                           print_labels,
                           prefix,
                           suffix,
                           print_perc,
                           filter_list=filter_regione,
                           level2=cfg.LEVEL_REGIONE,
                           labels_size=labels_size,
                           save_path=save_path,
                           dpi=dpi)


def plot_choropleth_map_provinciale(df,
                                    province_tag,
                                    value_tag,
                                    ax=None,
                                    title=None,
                                    color="b",
                                    show_colorbar=True,
                                    vmin=None,
                                    vmax=None,
                                    print_labels=False,
                                    prefix="",
                                    suffix="",
                                    filter_regione=None,
                                    filter_provincia=None,
                                    print_perc=False,
                                    labels_size=None,
                                    save_path=None,
                                    dpi=100):
    if filter_regione:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regione
    elif filter_provincia:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_provincia
    else:
        level_filter = None
        filter_list = None
    _create_choropleth_map(df,
                           province_tag,
                           value_tag,
                           cfg.LEVEL_PROVINCIA,
                           color,
                           ax,
                           title,
                           show_colorbar,
                           vmin,
                           vmax,
                           print_labels,
                           prefix,
                           suffix,
                           print_perc,
                           filter_list=filter_list,
                           level2=level_filter,
                           labels_size=labels_size,
                           save_path=save_path,
                           dpi=dpi)


def plot_choropleth_map_comunale(df,
                                 comuni_tag,
                                 value_tag,
                                 ax=None,
                                 title=None,
                                 color="b",
                                 show_colorbar=True,
                                 vmin=None,
                                 vmax=None,
                                 print_labels=False,
                                 prefix="",
                                 suffix="",
                                 filter_regione=None,
                                 filter_provincia=None,
                                 filter_comune=None,
                                 print_perc=False,
                                 labels_size=None,
                                 save_path=None,
                                 dpi=100):
    if filter_regione:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regione
    elif filter_provincia:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_provincia
    elif filter_comune:
        level_filter = cfg.LEVEL_COMUNE
        filter_list = filter_comune
    else:
        level_filter = None
        filter_list = None
    _create_choropleth_map(df,
                           comuni_tag,
                           value_tag,
                           cfg.LEVEL_COMUNE,
                           color,
                           ax,
                           title,
                           show_colorbar,
                           vmin,
                           vmax,
                           print_labels,
                           prefix,
                           suffix,
                           print_perc,
                           filter_list=filter_list,
                           level2=level_filter,
                           labels_size=labels_size,
                           save_path=save_path,
                           dpi=dpi)


def _create_choropleth_map_interactive(df0,
                                       geo_tag_input,
                                       dict_values,
                                       level,
                                       title,
                                       filter_list=None,
                                       level2=None):
    filter_list = _check_filter(filter_list)
    df = df0.copy()
    df = df[df[geo_tag_input].notnull()]

    code = _code_or_desc(list(df[geo_tag_input].unique()))

    shape = _get_shape_from_level(level)

    geo_tag_anag = _get_tag_anag(code, level)

    geoInf = AddGeographicalInfo(df)
    if level == cfg.LEVEL_COMUNE:
        geoInf.set_comuni_tag(geo_tag_input)
    elif level == cfg.LEVEL_PROVINCIA:
        geoInf.set_province_tag(geo_tag_input)
    elif level == cfg.LEVEL_REGIONE:
        geoInf.set_regioni_tag(geo_tag_input)

    geoInf.run_simple_match()
    if level == cfg.LEVEL_COMUNE:
        geoInf.run_find_frazioni()
    df = geoInf.get_result()
    del geoInf

    if filter_list is not None:
        code_filter = _code_or_desc(filter_list)
        tag_2 = _get_tag_anag(code_filter, level2)
        if code_filter == cfg.CODE_CODICE_ISTAT:
            filter_list = [int(x) for x in filter_list]
        elif code_filter == cfg.CODE_SIGLA:
            filter_list = [x.upper() for x in filter_list]
        else:
            filter_list = [_clean_denom_text_value(x) for x in filter_list]
        shape = shape[_clean_denom_text(shape[tag_2]).isin(filter_list)]

    col_list = list(dict_values.keys())
    col_list.append(geo_tag_anag)
    df = shape.merge(df[col_list], how="left", on=geo_tag_anag, suffixes=["_new", ""])

    for col in dict_values.keys():
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("-")

    shape_list = []
    if level == cfg.LEVEL_COMUNE:
        shape_province = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        shape_province = shape_province[shape_province[cfg.TAG_PROVINCIA].isin(shape[cfg.TAG_PROVINCIA].unique())]
        shape_province = gpd.GeoDataFrame(shape_province, geometry="geometry")
        shape_list.append((shape_province, 0.25))
        shape_regioni = _get_shape_from_level(cfg.LEVEL_REGIONE)
        shape_regioni = shape_regioni[shape_regioni[cfg.TAG_REGIONE].isin(shape[cfg.TAG_REGIONE].unique())]
        shape_regioni = gpd.GeoDataFrame(shape_regioni, geometry="geometry")
        shape_list.append((shape_regioni, 0.5))
    elif level == cfg.LEVEL_PROVINCIA:
        shape_regioni = _get_shape_from_level(cfg.LEVEL_REGIONE)
        shape_regioni = shape_regioni[shape_regioni[cfg.TAG_REGIONE].isin(shape[cfg.TAG_REGIONE].unique())]
        shape_regioni = gpd.GeoDataFrame(shape_regioni, geometry="geometry")
        shape_list.append((shape_regioni, 0.5))

    plot = plot_bokeh_choropleth_map(df, geo_tag_anag, level, dict_values, title=title, shape_list=shape_list)

    return plot


def plot_choropleth_map_comunale_interactive(df_comunale,
                                             comuni_tag,
                                             dict_values,
                                             title="",
                                             filter_regione=None,
                                             filter_provincia=None,
                                             filter_comune=None,
                                             save_path=None):
    if filter_regione:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regione
    elif filter_provincia:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_provincia
    elif filter_comune:
        level_filter = cfg.LEVEL_COMUNE
        filter_list = filter_comune
    else:
        level_filter = None
        filter_list = None
    plot = _create_choropleth_map_interactive(df_comunale,
                                              comuni_tag,
                                              dict_values,
                                              cfg.LEVEL_COMUNE,
                                              title=title,
                                              filter_list=filter_list,
                                              level2=level_filter)
    if save_path:
        output_file(save_path, mode='inline')
        save(plot)
        os.startfile(save_path)
    else:
        show(plot)


def plot_choropleth_map_provinciale_interactive(df_provinciale,
                                                province_tag,
                                                dict_values,
                                                title="",
                                                filter_regione=None,
                                                filter_provincia=None,
                                                save_path=None):
    if filter_regione:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regione
    elif filter_provincia:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_provincia
    else:
        level_filter = None
        filter_list = None
    plot = _create_choropleth_map_interactive(df_provinciale,
                                              province_tag,
                                              dict_values,
                                              cfg.LEVEL_PROVINCIA,
                                              title=title,
                                              filter_list=filter_list,
                                              level2=level_filter)
    if save_path:
        output_file(save_path, mode='inline')
        save(plot)
        os.startfile(save_path)
    else:
        show(plot)


def plot_choropleth_map_regionale_interactive(df_regionale,
                                              regioni_tag,
                                              dict_values,
                                              title="",
                                              filter_regione=None,
                                              save_path=None):
    if filter_regione:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regione
    else:
        level_filter = None
        filter_list = None
    plot = _create_choropleth_map_interactive(df_regionale,
                                              regioni_tag,
                                              dict_values,
                                              cfg.LEVEL_REGIONE,
                                              title=title,
                                              filter_list=filter_list,
                                              level2=level_filter)
    if save_path:
        output_file(save_path, mode='inline')
        save(plot)
        os.startfile(save_path)
    else:
        show(plot)


def plot_bokeh_choropleth_map(df0, geo_tag, level, dict_values, title="", shape_list=[]):
    geodf = gpd.GeoDataFrame(df0)

    inverted_dict = {value: key for (key, value) in dict_values.items()}
    field_list = list(dict_values.keys())
    n = len(field_list)

    palette_list_numerical = [Blues9,
                              Greens9,
                              Reds9,
                              Greys9,
                              Purples9,
                              Oranges9]
    palette_list = {}
    legend_list = {}
    is_numeric = {}
    i = 0
    for key, value in dict_values.items():
        if is_numeric_dtype(df0[key]):
            palette_list[key] = {"field": "values_plot", "transform": LinearColorMapper(
                palette=palette_list_numerical[i % len(palette_list_numerical)][::-1])}
            legend_list[key] = 0
            is_numeric[key] = True
            i += 1
        elif is_string_dtype(df0[key]):
            values = list(df0[key].unique())
            n_values = len(values)
            palette_list[key] = {"field": "values_plot", "transform": CategoricalColorMapper(factors=values,
                                                                                             palette=Category20[
                                                                                                 20] if n_values > 10 else
                                                                                             Category10[10])}
            is_numeric[key] = False
            legend_list[key] = 1
    geodf["values_plot"] = geodf[field_list[0]]
    geodf["line_color"] = "gray"
    geosource = GeoJSONDataSource(geojson=geodf.to_json())

    mapper = palette_list[field_list[0]]
    p = figure(title=title,
               plot_height=900,
               plot_width=800,
               tools='pan, wheel_zoom, box_zoom, reset')
    p.title.text_font_size = "25px"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

    fmt = NumberFormatter(format="0.[0] a")
    columns = [TableColumn(field=a, title=b, formatter=fmt) if is_numeric[a] else TableColumn(field=a, title=b) for a, b
               in dict_values.items()]
    # Add patch renderer to figure.
    if level == cfg.LEVEL_COMUNE:
        line_width = 0.1
        columns = [TableColumn(field=cfg.TAG_COMUNE, title=HEADER_BOKEH[cfg.LEVEL_COMUNE])] + columns
    elif level == cfg.LEVEL_PROVINCIA:
        line_width = 0.25
        columns = [TableColumn(field=cfg.TAG_PROVINCIA, title=HEADER_BOKEH[cfg.LEVEL_PROVINCIA])] + columns
    elif level == cfg.LEVEL_REGIONE:
        line_width = 0.5
        columns = [TableColumn(field=cfg.TAG_REGIONE, title=HEADER_BOKEH[cfg.LEVEL_REGIONE])] + columns
    else:
        line_width = 0.1

    data_table = DataTable(source=geosource, columns=columns, selectable=False)
    image = p.patches('xs', 'ys', source=geosource,
                      fill_color=mapper,
                      fill_alpha=0.7,
                      # line_color='gray',
                      line_color='line_color',
                      line_width=line_width)
    line_color = "darkgray"
    for shape, lw in shape_list:
        shape = GeoJSONDataSource(geojson=shape.to_json())
        p.patches('xs', 'ys', source=shape,
                  fill_alpha=0,
                  line_color=line_color,
                  line_width=lw)
        line_color = "black"

    tool_list = [(HEADER_BOKEH[level], '@' + geo_tag)]
    for key, values in dict_values.items():
        if is_numeric[key]:
            tool_list.append((values, '@' + key + '{0.[0] a}'))
        else:
            tool_list.append((values, '@' + key))
    p.add_tools(HoverTool(renderers=[image],
                          tooltips=tool_list))
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    color_bar = ColorBar(color_mapper=mapper["transform"], formatter=NumeralTickFormatter(format="0.[0] a"))
    p.add_layout(color_bar, 'right')

    if legend_list[field_list[0]] != 0:
        color_bar.visible = False

        factors = mapper["transform"].factors
        palette = mapper["transform"].palette
        colors = [palette[i % len(palette)] for i in range(len(factors))]
        data = {'x': [df0["geometry"].values[0].centroid.coords[0][0] for f in factors],
                'y': [df0["geometry"].values[0].centroid.coords[0][1] for f in factors],
                'color': colors,
                'factor': factors}
        source_legend = ColumnDataSource(data=data)

    else:
        source_legend = ColumnDataSource(data={"x": [],
                                               "y": [],
                                               "color": [],
                                               "factor": []})
    legend_image = p.circle(x="x", y="y", size=0, fill_color="color", legend="factor", line_width=0,
                            source=source_legend)
    p.legend.title = list(dict_values.values())[0]
    p.legend.title_text_font_size = "20px"
    p.legend.title_text_font_style = "bold"
    if legend_list[field_list[0]] == 0:
        p.legend.border_line_width = 0

    geosource.selected.js_on_change('indices', CustomJS(
        args=dict(source=geosource),
        code="""
            var f = cb_obj.indices[0];
            console.log(f);
            var data = source.data;
            data["line_color"][f] = "blue";
            source.change.emit();
            """
    ))

    if len(field_list) > 1:
        field_select = Select(title="Select:", value=list(dict_values.values())[0], options=list(dict_values.values()))
        callback_code = """
        var data = source.data;
        var data_legend = source_legend.data;
        var value_selected = inverted_dict[selection.value];
        var type = legend_list[value_selected];
        data['values_plot'] = data[value_selected];
        image.glyph.fill_color = palette_list[value_selected];
        legend[0].title = selection.value;
        for (var key in data_legend) {
                data_legend[key] = [];
                }
        if (type == 0){
            color_bar.visible=true;
            color_bar.color_mapper = palette_list[value_selected]["transform"];
            legend[0].border_line_width = 0;
            } else {
            var factors = palette_list[value_selected]["transform"].factors;
            var n_factor = factors.length;
            var palette = palette_list[value_selected]["transform"].palette;
            var colors = [];
            var xx = [];
            var yy = [];
            for (var i = 0; i < n_factor; ++i){
                colors.push(palette_list[value_selected]["transform"].palette[i%(palette.length)]);
                xx.push(data["center_x"][0]);
                yy.push(data["center_y"][0]);
            }
            data_legend["x"].push(...xx);
            data_legend["y"].push(...yy);
            data_legend["factor"].push(...factors);
            data_legend["color"].push(...colors);
            color_bar.visible=false;
            legend[0].border_line_width = 1;
        }
        console.log("Select: " + value_selected);
        source.change.emit();
        source_legend.change.emit();
        legend_image.change.emit();
        p.change.emit();
        """

        callback = CustomJS(
            args=dict(source=geosource,
                      source_legend=source_legend,
                      selection=field_select,
                      inverted_dict=inverted_dict,
                      palette_list=palette_list,
                      legend_list=legend_list,
                      color_bar=color_bar,
                      image=image,
                      legend_image=legend_image,
                      legend=p.legend,
                      p=p),
            code=callback_code
        )
        field_select.js_on_change("value", callback)

        plot = row(p, column(field_select, data_table))
    else:
        plot = row(p, data_table)
    return plot


def plot_point_map(df0,
                   latitude_columns=None,
                   longitude_columns=None,
                   filter_comune=None,
                   filter_provincia=None,
                   filter_regione=None,
                   color_tag=None,
                   ax=None,
                   title=None,
                   legend_font=None,
                   show_colorbar=True,
                   size=6,
                   save_in_path=None,
                   dpi=100):
    filter_comune = _check_filter(filter_comune)
    filter_provincia = _check_filter(filter_provincia)
    filter_regione = _check_filter(filter_regione)

    df = df0.copy()
    if (latitude_columns is None) or (longitude_columns is None):
        flag_coord_found, latitude_columns, longitude_columns = __find_coord_columns(df)

    df[latitude_columns] = df[latitude_columns].astype(float)
    df[longitude_columns] = df[longitude_columns].astype(float)
    coord_system_input = __find_coordinates_system(df, lat=latitude_columns, lon=longitude_columns)

    shape_list = []
    if filter_regione:
        polygon_df = get_df_regioni()
        polygon_df = polygon_df[polygon_df[cfg.TAG_REGIONE].isin(filter_regione)][["geometry"]]
        polygon_df = gpd.GeoDataFrame(polygon_df, geometry="geometry")
        polygon_df.crs = {'init': "epsg:32632"}
        polygon_df = polygon_df.to_crs({'init': coord_system_input})
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[longitude_columns], df[latitude_columns]))
        df = gpd.tools.sjoin(df, polygon_df, op='within')
        shape_list.append((polygon_df, 0.4, "0.6"))
        shape = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        shape = shape[shape[cfg.TAG_REGIONE].isin(filter_regione)]
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.2, "0.8"))
    elif filter_provincia:
        polygon_df = get_df_province()
        polygon_df = polygon_df[polygon_df[cfg.TAG_PROVINCIA].isin(filter_provincia)][["geometry"]]
        polygon_df = gpd.GeoDataFrame(polygon_df, geometry="geometry")
        polygon_df.crs = {'init': "epsg:32632"}
        polygon_df = polygon_df.to_crs({'init': coord_system_input})
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[longitude_columns], df[latitude_columns]))
        df = gpd.tools.sjoin(df, polygon_df, op='within')
        shape_list.append((polygon_df, 0.4, "0.6"))
        shape = _get_shape_from_level(cfg.LEVEL_COMUNE)
        shape = shape[shape[cfg.TAG_PROVINCIA].isin(filter_provincia)]
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.2, "0.8"))
    elif filter_comune:
        polygon_df = get_df_comuni()
        polygon_df = polygon_df[polygon_df[cfg.TAG_COMUNE].isin(filter_comune)][["geometry"]]
        polygon_df = gpd.GeoDataFrame(polygon_df, geometry="geometry")
        polygon_df.crs = {'init': "epsg:32632"}
        polygon_df = polygon_df.to_crs({'init': coord_system_input})
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[longitude_columns], df[latitude_columns]))
        df = gpd.tools.sjoin(df, polygon_df, op='within')
        shape_list.append((polygon_df, 0.4, "0.6"))
    else:
        shape = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.2, "0.8"))
        shape = _get_shape_from_level(cfg.LEVEL_REGIONE)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.4, "0.6"))

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    if title:
        ax.set_title(title)

    if color_tag:
        if is_numeric_dtype(df[color_tag]):
            vmin = df[color_tag].min()
            vmax = df[color_tag].max()
            vmin -= (vmax - vmin)/5
            cmap = get_cmap("Blues")
            ax.scatter(df[longitude_columns], df[latitude_columns], c=df[color_tag], cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       alpha=0.5, linewidths=0.1, s=size, edgecolors="blue")
            if show_colorbar:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm._A = []
                cbar = fig.colorbar(sm)
                if legend_font is None:
                    legend_font = 12
                cbar.ax.tick_params(labelsize=legend_font)
        elif is_string_dtype(df[color_tag]):
            color_labels = list(df[color_tag].unique())
            n_color = len(list(color_labels))
            if n_color <= 10:
                rgb_values = get_cmap("tab10")
                rgb_values = [rgb_values(i) for i in range(10)]
            else:
                rgb_values = get_cmap("tab20")
                rgb_values = [rgb_values(i) for i in range(20)]

            color_map = dict(zip(color_labels, rgb_values))
            for c in color_labels:
                df_plot = df[df[color_tag]==c]
                ax.scatter(df_plot[longitude_columns], df_plot[latitude_columns], color=color_map[c], label=c, alpha=0.5, linewidths=0.5,
                       s=size)
            if legend_font:
                ax.legend(loc="center left", title=color_tag, prop={'size': legend_font}, title_fontsize=legend_font*1.1,
                          bbox_to_anchor=(1, 0.5))
            else:
                ax.legend(loc="center left", title=color_tag, bbox_to_anchor=(1, 0.5))

    else:
        ax.scatter(df[longitude_columns], df[latitude_columns], c='blue', alpha=0.5, s=size)

    for shape, lw, ec in shape_list:
        shape.plot(facecolor="none", linewidth=lw, edgecolor=ec, ax=ax)
    ax.axis('off')
    if save_in_path:
        plt.savefig(save_in_path, bbox_inches='tight', dpi=dpi)
    else:
        if fig is not None:
            plt.show()
    return ax


def plot_point_map_interactive(df0,
                               latitude_columns=None,
                               longitude_columns=None,
                               filter_comune=None,
                               filter_provincia=None,
                               filter_regione=None,
                               color_tag=None,
                               info_dict=None,
                               title=None,
                               table=True,
                               plot_width=1500,
                               plot_height=800,
                               save_in_path=None,
                               show_flag=True):
    filter_comune = _check_filter(filter_comune)
    filter_provincia = _check_filter(filter_provincia)
    filter_regione = _check_filter(filter_regione)
    margins, shape = _get_margins(filter_comune=filter_comune,
                                  filter_provincia=filter_provincia,
                                  filter_regione=filter_regione)

    tile_provider = get_provider(CARTODBPOSITRON)

    plot = figure(x_range=(margins[0][0], margins[0][1]),
                  y_range=(margins[1][0], margins[1][1]),
                  x_axis_type="mercator", y_axis_type="mercator", plot_width=plot_width, plot_height=plot_height)
    plot.add_tile(tile_provider)

    if title is not None:
        plot.title.text = title
        plot.title.align = 'center'

    column_list = list(df0.columns)

    if (latitude_columns is None) or (longitude_columns is None):
        flag_coord_found, latitude_columns, longitude_columns = __find_coord_columns(df0)

    df = __create_geo_dataframe(df0, lat_tag=latitude_columns, long_tag=longitude_columns)
    if latitude_columns is None:
        latitude_columns = "geo_ita_lat"
        longitude_columns = "geo_ita_lon"
        df[latitude_columns] = df.geometry.y
        df[longitude_columns] = df.geometry.x
    df = df.to_crs({'init': 'epsg:3857'})
    if filter_regione or filter_comune or filter_provincia:
        df = gpd.tools.sjoin(df, shape, op='within')

    if info_dict != None:
        table_columns = list(info_dict.keys())
    else:
        table_columns = column_list
        if "geometry" in table_columns:
            table_columns.remove("geometry")
    if latitude_columns not in table_columns:
        table_columns.append(latitude_columns)
    if longitude_columns not in table_columns:
        table_columns.append(longitude_columns)

    df['x'] = df.geometry.x
    df['y'] = df.geometry.y
    df = pd.DataFrame(df.drop(columns='geometry'))

    source = ColumnDataSource(df)
    if info_dict is not None:
        columns = [TableColumn(field=a, title=b) for a, b in info_dict.items()]
    else:
        columns = [TableColumn(field=a, title=a) for a in table_columns if a not in [longitude_columns, latitude_columns]]

    legend = False
    if color_tag is not None:
        if is_numeric_dtype(df[color_tag]):
            min_values = df[color_tag].min()
            max_values = df[color_tag].max()
            if min_values >= 0:
                exp_cmap = LinearColorMapper(palette=Reds9[::-1], low=0,
                                             high=max_values)
            else:
                palette_max = max(np.abs(min_values), max_values)
                exp_cmap = LinearColorMapper(palette=RdYlGn11[::-1], low=-palette_max,
                                             high=palette_max)
            fill_color = {'field': color_tag, 'transform': exp_cmap}
        elif is_string_dtype(df[color_tag]):
            values = list(df[color_tag].unique())
            n_values = len(values)
            fill_color = {"field": color_tag, "transform": CategoricalColorMapper(factors=values,
                                                                                 palette=Category20[
                                                                                     20] if n_values > 10 else
                                                                                 Category10[10])}
            legend = True
        else:
            fill_color = "lime"
    else:
        fill_color = "lime"

    if legend:
        plot1 = plot.circle(x="x", y="y", size=7, fill_color=fill_color, line_width=0.5,
                            legend=color_tag,
                            source=source)
        plot.legend.title = color_tag
        plot.legend.title_text_font_size = "20px"
        plot.legend.title_text_font_style = "bold"
    else:
        plot1 = plot.circle(x="x", y="y", size=7, fill_color=fill_color, line_width=0.5,
                            source=source)

    tooltips1 = []
    if info_dict is not None:
        for key, values in info_dict.items():
            if is_numeric_dtype(df[key]):
                tooltips1.append((values, '@' + key + '{0.[0] a}'))
            else:
                tooltips1.append((values, '@' + key))
    else:
        for values in table_columns:
            if values not in [longitude_columns, latitude_columns]:
                if is_numeric_dtype(df[values]):
                    tooltips1.append((values, '@' + values + '{0.[0] a}'))
                else:
                    tooltips1.append((values, '@' + values))
    tooltips1.append(("Coords", "(@" + latitude_columns + "{0,0.0000000}-@" + longitude_columns + "{0,0.0000000})"))

    plot.add_tools(HoverTool(renderers=[plot1], tooltips=tooltips1))

    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
    if table:
        data_table = DataTable(source=source, columns=columns)
        p = row(plot, data_table)
    else:
        p = plot
    if save_in_path is not None:
        output_file(save_in_path, mode='inline')
        save(p)
    if show_flag:
        if save_in_path is None:
            show(p)
        else:
            os.startfile(save_in_path)
    return p


def _get_margins(filter_comune=None,
                 filter_provincia=None,
                 filter_regione=None):
    filter_comune = _check_filter(filter_comune)
    filter_provincia = _check_filter(filter_provincia)
    filter_regione = _check_filter(filter_regione)
    if filter_comune is not None:
        filter_comune = [_clean_denom_text_value(a) for a in filter_comune]
        code = _code_or_desc(filter_comune)
        shape = _get_shape_from_level(cfg.LEVEL_COMUNE)
        tag_shape = _get_tag_anag(code, cfg.LEVEL_COMUNE)
        shape[tag_shape] = _clean_denom_text(shape[tag_shape])
        margins = shape[shape[tag_shape].isin(filter_comune)]
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
    elif filter_provincia is not None:
        filter_provincia = [_clean_denom_text_value(a) for a in filter_provincia]
        code = _code_or_desc(filter_provincia)
        shape = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        tag_shape = _get_tag_anag(code, cfg.LEVEL_PROVINCIA)
        shape[tag_shape] = _clean_denom_text(shape[tag_shape])
        margins = shape[shape[tag_shape].isin(filter_provincia)]
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
    elif filter_regione is not None:
        filter_regione = [_clean_denom_text_value(a) for a in filter_regione]
        code = _code_or_desc(filter_regione)
        shape = _get_shape_from_level(cfg.LEVEL_REGIONE)
        tag_shape = _get_tag_anag(code, cfg.LEVEL_REGIONE)
        shape[tag_shape] = _clean_denom_text(shape[tag_shape])
        margins = shape[shape[tag_shape].isin(filter_regione)]
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
    else:
        margins = _get_shape_from_level(cfg.LEVEL_REGIONE)
        margins["key"] = "Italia"
        margins = gpd.GeoDataFrame(margins, geometry="geometry")
        margins = margins.dissolve(by='key')
    if len(margins) == 0:
        raise Exception("Unable to find the filter.")
    else:
        margins = margins[["geometry"]]
        margins.crs = {'init': "epsg:32632"}
        margins = margins.to_crs({'init': 'epsg:3857'})
        margins_coord = margins["geometry"].values
        margins_coord = (min([margins_coord[i].bounds[0] for i in range(len(margins_coord))]),
                       min([margins_coord[i].bounds[1] for i in range(len(margins_coord))]),
                       max([margins_coord[i].bounds[2] for i in range(len(margins_coord))]),
                       max([margins_coord[i].bounds[3] for i in range(len(margins_coord))]))
        margins_coord = [[margins_coord[0], margins_coord[2]], [margins_coord[1], margins_coord[3]]]

    #inProj, outProj = Proj(init='epsg:32632'), Proj(init='epsg:3857')
    #margins_coord2 = [[], []]
    #margins_coord2[0], margins_coord2[1] = transform(inProj, outProj, margins_coord[0], margins_coord[1])

    return margins_coord, margins


def _filter_margins(df, margins, long_tag=None, lat_tag=None):
    if long_tag:
        result = df[(df[long_tag] >= margins[0][0]) &
                    (df[long_tag] <= margins[0][1]) &
                    (df[lat_tag] >= margins[1][0]) &
                    (df[lat_tag] <= margins[1][1])
                ]
    else:
        result = df[(df.geometry.x >= margins[0][0]) &
                (df.geometry.x <= margins[0][1]) &
                (df.geometry.y >= margins[1][0]) &
                (df.geometry.y <= margins[1][1])
                ]
    return result


def plot_kernel_density_estimation(df0,
                                   latitude_columns=None,
                                   longitude_columns=None,
                                   value_tag=None,
                                   comune=None,
                                   provincia=None,
                                   regione=None,
                                   n_grid_x=1000,
                                   n_grid_y=1000,
                                   ax=None,
                                   title=None,
                                   save_in_path=None,
                                   dpi=100):
    df = df0.copy()
    if (latitude_columns is None) or (longitude_columns is None):
        flag_coord_found, latitude_columns, longitude_columns = __find_coord_columns(df)
    df[latitude_columns] = df[latitude_columns].astype(float)
    df[longitude_columns] = df[longitude_columns].astype(float)
    coord_system_input = __find_coordinates_system(df, lat=latitude_columns, lon=longitude_columns)

    shape_list = []

    if regione:
        polygon_df = get_df_regioni()
        polygon_df = polygon_df[polygon_df[cfg.TAG_REGIONE] == regione][["geometry"]]
        polygon_df = gpd.GeoDataFrame(polygon_df, geometry="geometry")
        polygon_df.crs = {'init': "epsg:32632"}
        polygon_df = polygon_df.to_crs({'init': coord_system_input})
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[longitude_columns], df[latitude_columns]))
        df = gpd.tools.sjoin(df, polygon_df, op='within')
        shape_list.append((polygon_df, 0.4, "0.6"))
        shape = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        shape = shape[shape[cfg.TAG_REGIONE] == regione]
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.2, "0.8"))
    elif provincia:
        polygon_df = get_df_province()
        polygon_df = polygon_df[polygon_df[cfg.TAG_PROVINCIA] == provincia][["geometry"]]
        polygon_df = gpd.GeoDataFrame(polygon_df, geometry="geometry")
        polygon_df.crs = {'init': "epsg:32632"}
        polygon_df = polygon_df.to_crs({'init': coord_system_input})
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[longitude_columns], df[latitude_columns]))
        df = gpd.tools.sjoin(df, polygon_df, op='within')
        shape_list.append((polygon_df, 0.4, "0.6"))
        shape = _get_shape_from_level(cfg.LEVEL_COMUNE)
        shape = shape[shape[cfg.TAG_PROVINCIA] == provincia]
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.2, "0.8"))
    elif comune:
        polygon_df = get_df_comuni()
        polygon_df = polygon_df[polygon_df[cfg.TAG_COMUNE] == comune][["geometry"]]
        polygon_df = gpd.GeoDataFrame(polygon_df, geometry="geometry")
        polygon_df.crs = {'init': "epsg:32632"}
        polygon_df = polygon_df.to_crs({'init': coord_system_input})
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[longitude_columns], df[latitude_columns]))
        df = gpd.tools.sjoin(df, polygon_df, op='within')
        shape_list.append((polygon_df, 0.4, "0.6"))
    else:
        shape = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.2, "0.8"))
        shape = _get_shape_from_level(cfg.LEVEL_REGIONE)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': "epsg:32632"}
        shape = shape.to_crs({'init': coord_system_input})
        shape_list.append((shape, 0.4, "0.6"))

    x, y = df[longitude_columns].values, df[latitude_columns].values
    x0, y0 = df[longitude_columns].min(), df[latitude_columns].min()
    x1, y1 = df[longitude_columns].max(), df[latitude_columns].max()

    if value_tag:
        weights = df[value_tag].clip(0.00001, None).values
    else:
        weights = np.ones(df.shape[0])

    weights = weights / weights.sum() * 10000

    h, _, _ = np.histogram2d(x, y, bins=(np.linspace(x0, x1, n_grid_x), np.linspace(y0, y1, n_grid_y)), weights=weights)
    h[h == 0] = 1

    z = scipy.ndimage.filters.gaussian_filter(np.log(h.T), 1)

    z[z <= 0] = np.nan

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(z, origin='lower', extent=[x0, x1, y0, y1], cmap=get_cmap('Reds'))
    ax.axis('off')
    if title:
        ax.set_title(title)

    for shape, lw, ec in shape_list:
        shape.plot(facecolor="none", linewidth=lw, edgecolor=ec, ax=ax)
    if save_in_path:
        plt.savefig(save_in_path, bbox_inches='tight', dpi=dpi)
    else:
        if fig is not None:
            plt.show()
    return ax


def plot_kernel_density_estimation_interactive(df0,
                                               latitude_columns=None,
                                               longitude_columns=None,
                                               value_tag=None,
                                               filter_comune=None,
                                               filter_provincia=None,
                                               filter_regione=None,
                                               n_grid_x=1000,
                                               n_grid_y=1000,
                                               title=None,
                                               plot_width=1500,
                                               plot_height=800,
                                               save_in_path=None,
                                               show_flag=True):
    df = df0.copy()
    if (latitude_columns is None) or (longitude_columns is None):
        flag_coord_found, latitude_columns, longitude_columns = __find_coord_columns(df)
    df[latitude_columns] = df[latitude_columns].astype(float)
    df[longitude_columns] = df[longitude_columns].astype(float)
    filter_comune = _check_filter(filter_comune)
    filter_provincia = _check_filter(filter_provincia)
    filter_regione = _check_filter(filter_regione)
    margins, shape = _get_margins(filter_comune=filter_comune,
                                  filter_provincia=filter_provincia,
                                  filter_regione=filter_regione)
    df = __create_geo_dataframe(df0, lat_tag=latitude_columns, long_tag=longitude_columns)
    df = df.to_crs({'init': 'epsg:3857'})
    if filter_regione or filter_comune or filter_provincia:
        df = gpd.tools.sjoin(df, shape, op='within')
    if latitude_columns is None:
        latitude_columns = "geo_ita_lat"
        longitude_columns = "geo_ita_lon"
        df[latitude_columns] = df.geometry.y
        df[longitude_columns] = df.geometry.x

    x0, y0 = df[longitude_columns].min(), df[latitude_columns].min()
    x1, y1 = df[longitude_columns].max(), df[latitude_columns].max()

    if value_tag:
        weights = df[value_tag].clip(0.00001, None).values
    else:
        weights = np.ones(df.shape[0])

    weights = weights / weights.max() * 10000

    h, _, _ = np.histogram2d(df[longitude_columns], df[latitude_columns], bins=(np.linspace(x0, x1, n_grid_x), np.linspace(y0, y1, n_grid_y)), weights=weights)
    h[h == 0] = 1

    z = scipy.ndimage.filters.gaussian_filter(np.log(h.T), 1)

    z[z <= 0] = 0

    tile_provider = get_provider(CARTODBPOSITRON)

    plot = figure(x_range=(margins[0][0], margins[0][1]),
                  y_range=(margins[1][0], margins[1][1]),
                  x_axis_type="mercator", y_axis_type="mercator", plot_width=plot_width, plot_height=plot_height)
    plot.add_tile(tile_provider)
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.yaxis.visible = False
    plot.grid.visible = False
    plot.toolbar.logo = None
    plot.outline_line_color = None
    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None
    plot.xaxis.major_label_text_font_size = '0pt'
    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
    if title is not None:
        plot.title.text = title
        plot.title.align = 'center'
    palette = list(Reds9[::-1])
    palette[0] = 'rgba(0, 0, 0, 0)'
    palette = tuple(palette)
    if df.shape[0] > 0:
        plot.image(image=[z],
                   x=x0, y=y0, dw=x1 - x0, dh=y1 - y0,
                   palette=palette, level="image", global_alpha=0.5)
    if save_in_path is not None:
        output_file(save_in_path, mode='inline')
        save(plot)
    if show_flag:
        if save_in_path is None:
            show(plot)
        else:
            os.startfile(save_in_path)
    return plot
