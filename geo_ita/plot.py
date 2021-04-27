import os
import logging

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import geopandas as gpd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from bokeh.palettes import RdYlGn11, RdYlBu11, Blues9, Greens9, Reds9, Greys9, Purples9, Oranges9, Category10, \
    Category20
from bokeh.plotting import output_file, save, figure
from bokeh.layouts import column, row
from bokeh.io import output_file, output_notebook, show
from bokeh.models.mappers import LinearColorMapper
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.models import (Title, ColumnDataSource, Circle,
                          WheelZoomTool, BoxSelectTool, InvertedTriangle, HoverTool, ImageURL, Legend, LegendItem,
                          DataTable, TableColumn, Select,
                          CustomJS, TextInput, GeoJSONDataSource, Diamond, Label, LabelSet, ColorBar,
                          CategoricalColorMapper)

import geo_ita.config as cfg
from geo_ita.data import create_df_comuni, create_df_province, create_df_regioni
from geo_ita.enrich_dataframe import _clean_denom_text_value, _clean_denom_text, _get_tag_anag, _code_or_desc, \
    AddGeographicalInfo, __create_geo_dataframe, __find_coord_columns
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


def _plot_choropleth_map(df, color, ax, show_colorbar, numeric_values, value_tag, line_width=0.8):
    if numeric_values:
        vmin, vmax = df["count"].min(), df["count"].max()
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
        if show_colorbar & numeric_values:
            cbar = fig.colorbar(sm)
            cbar.ax.tick_params(labelsize=10)
    ax.axis('off')
    if numeric_values:
        df.plot('count', cmap=cmap, vmin=vmin, vmax=vmax, linewidth=line_width, edgecolor='0.8', ax=ax)
    else:
        df.plot('count', linewidth=line_width, edgecolor='0.8', ax=ax, color=df["color"].values)
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
                                horizontalalignment='center', color='black', wrap=True)  # , fontsize='large'
    else:
        for idx, row in df.iterrows():
            ax.annotate(text=row['count'],
                        xy=(row['center_x'], row['center_y']),
                        horizontalalignment='center', color='black', wrap=True)


def _get_shape_from_level(level):
    if level == cfg.LEVEL_COMUNE:
        shape = create_df_comuni()
    elif level == cfg.LEVEL_PROVINCIA:
        shape = create_df_province()
    elif level == cfg.LEVEL_REGIONE:
        shape = create_df_regioni()
    else:
        raise Exception("Level UNKNOWN")
    return shape


def _create_choropleth_map(df0,
                           geo_tag_input,
                           value_tag,
                           level,
                           color,
                           ax,
                           show_colorbar,
                           print_labels,
                           print_perc,
                           filter_list=None,
                           level2=None):
    # Todo add unità di misura labels / clorobar
    # Todo Cambio nome legenda
    # Todo Set title, labels size
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

        df = df[_clean_denom_text(df[tag_2]).isin(filter_list)]

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
    if level == cfg.LEVEL_COMUNE:
        line_width = 0.2
    elif level == cfg.LEVEL_PROVINCIA:
        line_width = 0.4
    elif level == cfg.LEVEL_REGIONE:
        line_width = 0.8
    else:
        line_width = 0.2
    log.debug(shape.head(5))
    fig, ax = _plot_choropleth_map(shape, color, ax, show_colorbar, numeric_values, value_tag, line_width=line_width)

    if print_labels:
        _add_labels_on_plot(shape, ax, print_perc, numeric_values)

    if fig is not None:
        plt.show()

    return fig


def plot_choropleth_map_regionale(df,
                                  region_tag,
                                  value_tag,
                                  ax=None,
                                  color="b",
                                  show_colorbar=True,
                                  print_labels=True,
                                  filter_regioni=None,
                                  print_perc=True):
    _create_choropleth_map(df,
                           region_tag,
                           value_tag,
                           cfg.LEVEL_REGIONE,
                           color,
                           ax,
                           show_colorbar,
                           print_labels,
                           print_perc,
                           filter_list=filter_regioni,
                           level2=cfg.LEVEL_REGIONE)


def plot_choropleth_map_provinciale(df,
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
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regioni
    elif filter_province:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_province
    else:
        level_filter = None
        filter_list = None
    _create_choropleth_map(df,
                           province_tag,
                           value_tag,
                           cfg.LEVEL_PROVINCIA,
                           color,
                           ax,
                           show_colorbar,
                           print_labels,
                           print_perc,
                           filter_list=filter_list,
                           level2=level_filter)


def plot_choropleth_map_comunale(df,
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
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regioni
    elif filter_province:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_province
    elif filter_comuni:
        level_filter = cfg.LEVEL_COMUNE
        filter_list = filter_comuni
    else:
        level_filter = None
        filter_list = None
    _create_choropleth_map(df,
                           comuni_tag,
                           value_tag,
                           cfg.LEVEL_COMUNE,
                           color,
                           ax,
                           show_colorbar,
                           print_labels,
                           print_perc,
                           filter_list=filter_list,
                           level2=level_filter)


def _create_choropleth_map_interactive(df0,
                                       geo_tag_input,
                                       dict_values,
                                       level,
                                       title,
                                       filter_list=None,
                                       level2=None):
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

    plot = plot_bokeh_choropleth_map(df, geo_tag_anag, level, dict_values, title=title)

    return plot


def plot_choropleth_map_comunale_interactive(df_comunale,
                                             comuni_tag,
                                             dict_values,
                                             title="",
                                             filter_regioni=None,
                                             filter_province=None,
                                             filter_comuni=None,
                                             save_path=None):
    if filter_regioni:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regioni
    elif filter_province:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_province
    elif filter_comuni:
        level_filter = cfg.LEVEL_COMUNE
        filter_list = filter_comuni
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
                                                filter_regioni=None,
                                                filter_province=None,
                                                save_path=None):
    if filter_regioni:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regioni
    elif filter_province:
        level_filter = cfg.LEVEL_PROVINCIA
        filter_list = filter_province
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
                                              filter_regioni=None,
                                              save_path=None):
    if filter_regioni:
        level_filter = cfg.LEVEL_REGIONE
        filter_list = filter_regioni
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


def plot_bokeh_choropleth_map(df0, geo_tag, level, dict_values, title=""):
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
    i = 0
    for key, value in dict_values.items():
        if is_numeric_dtype(df0[key]):
            palette_list[key] = {"field": "values_plot", "transform": LinearColorMapper(
                palette=palette_list_numerical[i % len(palette_list_numerical)][::-1])}
            legend_list[key] = 0
            i += 1
        elif is_string_dtype(df0[key]):
            values = list(df0[key].unique())
            n_values = len(values)
            palette_list[key] = {"field": "values_plot", "transform": CategoricalColorMapper(factors=values,
                                                                                             palette=Category20[
                                                                                                 20] if n_values > 10 else
                                                                                             Category10[10])}
            legend_list[key] = 1
    geodf["values_plot"] = geodf[field_list[0]]
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

    # Add patch renderer to figure.
    if level == cfg.LEVEL_COMUNE:
        line_width = 0.1
    elif level == cfg.LEVEL_PROVINCIA:
        line_width = 0.24
    elif level == cfg.LEVEL_REGIONE:
        line_width = 0.5
    else:
        line_width = 0.1
    image = p.patches('xs', 'ys', source=geosource,
                      fill_color=mapper,
                      fill_alpha=0.7,
                      line_color='gray',
                      line_width=line_width)
    tool_list = [(HEADER_BOKEH[level], '@' + geo_tag)]
    for key, values in dict_values.items():
        tool_list.append((values, '@' + key))
    p.add_tools(HoverTool(renderers=[image],
                          tooltips=tool_list))
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    color_bar = ColorBar(color_mapper=mapper["transform"])
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
        if (type == 0){
            color_bar.visible=true;
            color_bar.color_mapper = palette_list[value_selected]["transform"];
            for (var key in data_legend) {
                data_legend[key] = [];
                }
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

        plot = row(p, column(field_select))
    else:
        plot = row(p)
    return plot


def plot_point_map_interactive(df0,
                               comune=None,
                               provincia=None,
                               regione=None,
                               columns_dict=None,
                               title=None,
                               save_in_path=None):
    margins = _get_margins(comune=comune,
                           provincia=provincia,
                           regione=regione)

    tile_provider = get_provider(CARTODBPOSITRON)

    plot = figure(x_range=(margins[0][0], margins[0][1]),
                  y_range=(margins[1][0], margins[1][1]),
                  x_axis_type="mercator", y_axis_type="mercator", plot_width=1500, plot_height=800)
    plot.add_tile(tile_provider)

    if title is not None:
        plot.title.text = title
        plot.title.align = 'center'

    column_list = list(df0.columns)
    flag_coord_found, lat_tag, long_tag = __find_coord_columns(df0)
    df = __create_geo_dataframe(df0)
    df = df.to_crs({'init': 'epsg:3857'})

    if columns_dict != None:
        table_columns = list(columns_dict.keys())
    else:
        table_columns = column_list
        table_columns.remove("geometry")
    if lat_tag not in table_columns:
        table_columns.append(lat_tag)
    if long_tag not in table_columns:
        table_columns.append(long_tag)

    df['x'] = df.geometry.x
    df['y'] = df.geometry.y
    df = pd.DataFrame(df.drop(columns='geometry'))

    source = ColumnDataSource(df)
    if columns_dict != None:
        columns = [TableColumn(field=a, title=b) for a, b in columns_dict.items()]
    else:
        columns = [TableColumn(field=a, title=a) for a in table_columns if a not in [long_tag, lat_tag]]

    data_table = DataTable(source=source, columns=columns)

    image1 = Circle(x="x", y="y",
                    size=7,
                    fill_color="lime",
                    line_color="green", line_width=0.5)
    plot1 = plot.add_glyph(source, image1)

    if columns_dict != None:
        tooltips1 = [(b, "@{" + a + "}") for a, b in columns_dict.items()]

    else:
        tooltips1 = [(a, "@{" + a + "}") for a in table_columns if a not in [long_tag, lat_tag]]
    tooltips1.append(("Coords", "(@" + lat_tag + "{0,0.0000000},@" + long_tag + "{0,0.0000000})"))

    plot.add_tools(HoverTool(renderers=[plot1], tooltips=tooltips1))

    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)

    p = row(plot, data_table)
    if save_in_path is None:
        show(p)
    else:
        output_file(save_in_path, mode='inline')
        save(p)
        os.startfile(save_in_path)


def _get_margins(comune=None,
                 provincia=None,
                 regione=None):
    if comune is not None:
        comune = _clean_denom_text_value(comune)
        code = _code_or_desc([comune])
        shape = _get_shape_from_level(cfg.LEVEL_COMUNE)
        tag_shape = _get_tag_anag(code, cfg.LEVEL_COMUNE)
        shape[tag_shape] = _clean_denom_text(shape[tag_shape])
        margins = shape.loc[shape[tag_shape] == comune, "geometry"].values
        if len(margins) > 0:
            margins = margins[0].bounds
            margins = [[margins[0], margins[2]], [margins[1], margins[3]]]
    elif provincia is not None:
        provincia = _clean_denom_text_value(provincia)
        code = _code_or_desc([provincia])
        shape = _get_shape_from_level(cfg.LEVEL_PROVINCIA)
        tag_shape = _get_tag_anag(code, cfg.LEVEL_PROVINCIA)
        shape[tag_shape] = _clean_denom_text(shape[tag_shape])
        margins = shape.loc[shape[tag_shape] == provincia, "geometry"].values
        if len(margins) > 0:
            margins = margins[0].bounds
            margins = [[margins[0], margins[2]], [margins[1], margins[3]]]
    elif regione is not None:
        regione = _clean_denom_text_value(regione)
        code = _code_or_desc([regione])
        shape = _get_shape_from_level(cfg.LEVEL_REGIONE)
        tag_shape = _get_tag_anag(code, cfg.LEVEL_REGIONE)
        shape[tag_shape] = _clean_denom_text(shape[tag_shape])
        margins = shape.loc[shape[tag_shape] == regione, "geometry"].values
        if len(margins) > 0:
            margins = margins[0].bounds
            margins = [[margins[0], margins[2]], [margins[1], margins[3]]]
    else:
        margins = cfg.plot_italy_margins_32632

    inProj, outProj = Proj(init='epsg:32632'), Proj(init='epsg:3857')
    margins_coord = [[], []]
    margins_coord[0], margins_coord[1] = transform(inProj, outProj, margins[0], margins[1])

    return margins_coord
