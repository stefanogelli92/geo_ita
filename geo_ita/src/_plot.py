import os
import logging
from typing import Union, Dict, List, Optional
from pathlib import Path

import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
from pandas.api.types import is_numeric_dtype, is_string_dtype
from bokeh.palettes import (
    Blues9, Greens9, Reds9, Greys9, Purples9, Oranges9, Category10, Category20, RdYlGn11, Viridis256
)
import contextily as ctx
from bokeh.plotting import save, figure
from bokeh.layouts import column, row
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import (
    ColumnDataSource, WheelZoomTool, HoverTool, DataTable, TableColumn, Select, CustomJS, GeoJSONDataSource, ColorBar,
    CategoricalColorMapper, NumberFormatter, NumeralTickFormatter, Plot
)
from valdec.decorators import validate
import xyzservices.providers as xyz

import geo_ita.src.config as cfg
from geo_ita.src._data import get_df
from geo_ita.src._data_enrichment import (
    _clean_denomination_text, AddGeographicalInfo, _create_geo_dataframe, _get_margins
)
from geo_ita.src.utils import infer_geographical_category, get_tag_registry, ensure_list, GeoLevel, CodeLevel, \
    clean_denomination_text_value, _linear_colormap, _human_format

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

PLOT_SUFFIX_COLUMNS = "_plot_suffix_addGeoInfo"
PLOT_VALUE_COLUMN = "geo_ita_value_plot"


@validate
def plot_choropleth_map_regionale(
    df: pd.DataFrame,
    regione_column: str,
    color_column: str,
    **kwargs,
):
    """
    Wrapper for regioni choropleth map plotting.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        regione_column (str): Column name in `df` with regione identifiers.
        color_column (str): Column name in `df` with values to visualize.
        kwargs: Additional arguments passed to the generic `plot_choropleth_map`.

    Returns:
        None
    """
    _create_choropleth_map(df, regione_column, color_column, GeoLevel.REGIONE, interactive=False, **kwargs)


@validate
def plot_choropleth_map_provinciale(
    df: pd.DataFrame,
    provincia_column: str,
    color_column: str,
    **kwargs,
):
    """
    Wrapper for province choropleth map plotting.

    Args:
       df (pd.DataFrame): DataFrame containing the data to plot.
       provincia_column (str): Column name in `df` with provincia identifiers.
       color_column (str): Column name in `df` with values to visualize.
       kwargs: Additional arguments passed to the generic `plot_choropleth_map`.

    Returns:
       None
    """
    _create_choropleth_map(df, provincia_column, color_column, GeoLevel.PROVINCIA, interactive=False, **kwargs)


@validate
def plot_choropleth_map_comunale(
    df: pd.DataFrame,
    comune_column: str,
    color_column: str,
    **kwargs
):
    """
    Wrapper for comuni choropleth map plotting.

    Args:
      df (pd.DataFrame): DataFrame containing the data to plot.
      comune_column (str): Column name in `df` with comune identifiers.
      color_column (str): Column name in `df` with values to visualize.
      kwargs: Additional arguments passed to the generic `plot_choropleth_map`.

    Returns:
      None
    """
    _create_choropleth_map(df, comune_column, color_column, GeoLevel.COMUNE, interactive=False, **kwargs)


@validate
def plot_choropleth_map_comunale_interactive(
    df: pd.DataFrame,
    comune_column: str,
    color_columns: Union[str, list, dict],
    **kwargs
):
    """
    Wrapper for comuni choropleth map plotting.

    Args:
      df (pd.DataFrame): DataFrame containing the data to plot.
      comune_column (str): Column name in `df` with comune identifiers.
      color_columns (Union[str, list, dict]): Columns name in `df` with values to visualize.
      kwargs: Additional arguments passed to the generic `plot_choropleth_map`.

    Returns:
      None
    """
    _create_choropleth_map(df, comune_column, color_columns, GeoLevel.COMUNE, interactive=True, **kwargs)


@validate
def plot_choropleth_map_provinciale_interactive(
    df: pd.DataFrame,
    provincia_column: str,
    color_columns: Union[str, list, dict],
    **kwargs
):
    """
    Wrapper for province choropleth map plotting.

    Args:
      df (pd.DataFrame): DataFrame containing the data to plot.
      provincia_column (str): Column name in `df` with provincia identifiers.
      color_columns (Union[str, list, dict]): Columns name in `df` with values to visualize.
      kwargs: Additional arguments passed to the generic `plot_choropleth_map`.

    Returns:
      None
    """
    _create_choropleth_map(df, provincia_column, color_columns, GeoLevel.PROVINCIA, interactive=True, **kwargs)


@validate
def plot_choropleth_map_regionale_interactive(
    df: pd.DataFrame,
    regione_column: str,
    color_columns: Union[str, list, dict],
    **kwargs
):
    """
    Wrapper for regioni choropleth map plotting.

    Args:
      df (pd.DataFrame): DataFrame containing the data to plot.
      regione_column (str): Column name in `df` with regione identifiers.
      color_columns (Union[str, list, dict]): Columns name in `df` with values to visualize.
      kwargs: Additional arguments passed to the generic `plot_choropleth_map`.

    Returns:
      None
    """
    _create_choropleth_map(df, regione_column, color_columns, GeoLevel.REGIONE, interactive=True, **kwargs)


def _create_choropleth_map(
    df0: pd.DataFrame,
    geo_column: str,
    color_columns: Union[str, list, dict],
    geo_level: GeoLevel,
    filter_regione: Optional[Union[str, List[str]]] = None,
    filter_provincia: Optional[Union[str, List[str]]] = None,
    filter_comune: Optional[Union[str, List[str]]] = None,
    fillna: Optional[Union[float, str, bool]] = 0,
    interactive: bool = False,
    aggregate_function: Optional[Union[str, list]] = None,
    **kwargs
):
    """
    Plots a choropleth map at the specified geographic level.

    Args:
       df0 (pd.DataFrame): DataFrame containing the data to plot.
       geo_column (str): Column name in `df` that contains the geographic identifiers.
       color_columns (Union[str, list, dict]): Column names in `df` that contains the values to be visualized.
       geo_level (GeoLevel): Geographic level to plot (e.g., REGIONE, PROVINCIA, COMUNE).
       filter_regione (Union[str, List[str]], optional): Filter for specific regions. Defaults to None.
       filter_provincia (Union[str, List[str]], optional): Filter for specific provinces. Defaults to None.
       filter_comune (Union[str, List[str]], optional): Filter for specific municipalities. Defaults to None.
       fillna (Union[float, str, bool], optional): Value to fill NaN values. Defaults to 0.
       interactive (bool, optional): Whether to create an interactive plot. Defaults to False.
       kwargs: Additional arguments passed to the generic `_plot_choropleth_map`.

    Returns:
       None
    """
    # Todo add unità di misura labels / clorobar
    # Todo Cambio nome legenda
    # Todo Plot backgroud regions grey
    # Todo auto check if center scale and use 3 color map

    df = _prepare_choropleth_data(df0, geo_column, geo_level)
    geo_column = get_tag_registry(CodeLevel.CODE, geo_level)

    shape = get_df(geo_level)

    filter_level, filter_list = _get_filter_params(filter_regione, filter_provincia, filter_comune)
    filter_list = ensure_list(filter_list)
    shape = _filter_data(shape, filter_list, filter_level)

    color_columns = color_columns if isinstance(color_columns, dict) else {v: v for v in ensure_list(color_columns)}

    col_list = list(color_columns.keys())
    col_list.append(geo_column)

    df = shape.merge(df[col_list], how="left", on=geo_column, suffixes=["_new", ""])

    for col in color_columns.keys():
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(fillna)
        else:
            df[col] = df[col].fillna("NaN")

    if df.groupby(geo_column).size().max() > 1:
        if aggregate_function is None:
            raise Exception("The plot has multiple values for the same geographical level. Elaborate the data "
                            "accordingly or pass an aggregate function to the plot.")
        else:
            aggregate_function = aggregate_function if isinstance(aggregate_function, list) \
                else [aggregate_function for _ in range(len(color_columns))]
            df = (
                df.groupby(geo_column)
                .agg({k: v for k, v in zip(color_columns.keys(), aggregate_function)})
                .reset_index()
            )

    df = gpd.GeoDataFrame(df, geometry="geometry")

    shape_list = get_boundaries_list(geo_level, df)

    line_width = 0.2 if geo_level in [GeoLevel.COMUNE, None] else 0.4 if geo_level == GeoLevel.PROVINCIA else 0.8

    if not interactive:
        return _plot_choropleth_map(df, list(color_columns.keys())[0], shape_list, line_width, **kwargs)
    else:
        return _plot_bokeh_choropleth_map(df, geo_level, color_columns, shape_list, **kwargs)


def _prepare_choropleth_data(df0, geo_column, geo_level):
    """
    Prepares the DataFrame for choropleth map plotting.
    """
    geo_inf = AddGeographicalInfo(df0)
    if geo_level == GeoLevel.COMUNE:
        geo_inf.set_comuni_tag(geo_column)
    elif geo_level == GeoLevel.PROVINCIA:
        geo_inf.set_province_tag(geo_column)
    elif geo_level == GeoLevel.REGIONE:
        geo_inf.set_regioni_tag(geo_column)
    geo_inf.run_simple_match()
    if geo_level == GeoLevel.COMUNE:
        geo_inf.run_find_frazioni()
        geo_inf.run_similarity_match(unique_flag=False)
        geo_inf.accept_similarity_result()
    df = geo_inf.get_result(handle_duplicate_column="overwrite", drop_not_match=True)
    return df


def _filter_data(shape, filter_list, filter_level):
    """
    Filters the shape DataFrame based on user input.
    """
    if filter_list is not None:
        code_filter = infer_geographical_category(filter_list)
        tag_filter = get_tag_registry(code_filter, filter_level)
        if code_filter == CodeLevel.CODE:
            filter_list = [int(x) for x in filter_list]
        elif code_filter == CodeLevel.SIGLA:
            filter_list = [x.lower() for x in filter_list]
        else:
            filter_list = [clean_denomination_text_value(x) for x in filter_list]
        shape = shape[_clean_denomination_text(shape[tag_filter]).isin(filter_list)]
    return shape


def _get_filter_params(filter_regione, filter_provincia, filter_comune):
    """
    Determines the appropriate filter list and level based on user input.
    """
    if filter_regione:
        return GeoLevel.REGIONE, filter_regione
    elif filter_provincia:
        return GeoLevel.PROVINCIA, filter_provincia
    elif filter_comune:
        return GeoLevel.COMUNE, filter_comune
    return None, None


def get_boundaries_list(geo_level, df):
    """
    Returns a list of shapes to be plotted on the map.
    """
    shape_list = []
    if geo_level == GeoLevel.COMUNE:
        shape_province = get_df(GeoLevel.PROVINCIA)
        shape_province = shape_province[shape_province[cfg.TAG_PROVINCIA].isin(df[cfg.TAG_PROVINCIA].unique())]
        shape_province = gpd.GeoDataFrame(shape_province, geometry="geometry")
        shape_list.append((shape_province, 0.4, GeoLevel.PROVINCIA))
        shape_regioni = get_df(GeoLevel.REGIONE)
        shape_regioni = shape_regioni[shape_regioni[cfg.TAG_REGIONE].isin(df[cfg.TAG_REGIONE].unique())]
        shape_regioni = gpd.GeoDataFrame(shape_regioni, geometry="geometry")
        shape_list.append((shape_regioni, 0.8, GeoLevel.REGIONE))
    elif geo_level == GeoLevel.PROVINCIA:
        shape_regioni = get_df(GeoLevel.REGIONE)
        shape_regioni = shape_regioni[shape_regioni[cfg.TAG_REGIONE].isin(df[cfg.TAG_REGIONE].unique())]
        shape_regioni = gpd.GeoDataFrame(shape_regioni, geometry="geometry")
        shape_list.append((shape_regioni, 0.8, GeoLevel.REGIONE))
    return shape_list


def _plot_choropleth_map(
    df: gpd.GeoDataFrame,
    color_column: str,
    boundary_list: List,
    line_width,
    color_scale: str = "blue",
    ax: Axes = None,
    title: str = None,
    title_size: Union[int, float] = None,
    show_colorbar: bool = True,
    min_value: float = None,
    max_value: float = None,
    value_tag: str = "value",
    print_labels: bool = True,
    prefix: str = "",
    suffix: str = "",
    labels_size: Union[int, float] = None,
    label_format: callable = _human_format,
    facecolor: Union[str, bool] = True,
    fig_size: tuple = (20, 10),
    save_path: Union[str, Path] = None,
    show_plot: bool = True,
    dpi: int = 100,
):
    """
    Plots a choropleth map with the specified parameters.

    Args:
         df (gpd.GeoDataFrame): DataFrame containing the data to plot.
         color_column (str): Column name in `df` with the values to visualize.
         boundary_list (List): List of shapes to be plotted on the map.
         line_width: Width of the boundary lines.
         color_scale (str): Name of the color to use. Defaults to "blue".
         ax (Axes): Matplotlib Axes object to plot on. Defaults to None.
         title (str): Title of the plot. Defaults to None.
         title_size (Union[int, float]): Font size for the title. Defaults to None.
         show_colorbar (bool): Whether to display the colorbar. Defaults to True.
         min_value (float): Minimum value for the colorbar. Defaults to None.
         max_value (float): Maximum value for the colorbar. Defaults to None.
         value_tag (str): Column name in `df` with the values to visualize. Defaults to "value".
         print_labels (bool): Whether to print the labels. Defaults to True.
         prefix (str): Prefix to add to the values. Defaults to "".
         suffix (str): Suffix to add to the values. Defaults to "".
         labels_size (Union[int, float]): Font size for the labels. Defaults to None.
         facecolor (Union[str, bool]): Background color of the plot. Defaults to True.
         fig_size (tuple): Size of the figure. Defaults to (20, 10).
         save_path (Union[str, Path]): Path to save the plot. Defaults to None.
         show_plot (bool): Whether to display the plot. Defaults to True.
         dpi (int): Resolution of the saved plot. Defaults to 100.
    """
    facecolor = "azure" if facecolor is True else "white" if facecolor is False else facecolor

    label_format_tot = lambda x: str(prefix) + label_format(x) + str(suffix)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, figsize=fig_size, facecolor=facecolor)
        if title:
            if (title_size is None) and (labels_size is None):
                title_size = 'large'
            elif (title_size is None) and (labels_size is not None):
                title_size = labels_size * 1.1
            ax.set_title(title, fontsize=title_size)
    ax.axis('off')

    if is_numeric_dtype(df[color_column]):
        if min_value is None:
            min_value = df[color_column].min()
        if max_value is None:
            max_value = df[color_column].max()
        cmap = _linear_colormap(color_name2=color_scale)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_value, vmax=max_value))
        sm._A = []
        df.plot(color_column, cmap=cmap, vmin=min_value, vmax=max_value, linewidth=line_width, edgecolor='0.8', ax=ax)
        if show_colorbar:
            fmt = lambda x, pos: label_format_tot(x)
            cbar = plt.colorbar(sm, format=FuncFormatter(fmt), ax=ax)
            cbar.ax.tick_params(labelsize=10)
    else:
        color_map = df[color_column].value_counts(dropna=False).reset_index()
        if color_map.shape[0] <= 10:
            cmap = get_cmap('tab10')
        else:
            cmap = get_cmap('tab20')
        color_map["color"] = None
        for i in range(color_map.shape[0]):
            color_map.iat[i, 2] = cmap(i % 10)
        df["color"] = df[color_column].map(color_map.set_index("index")["color"])
        legend_elements = [Patch(facecolor=row.color, label=row["index"]) for index, row in color_map.iterrows()]
        df.plot(color_column, linewidth=line_width, edgecolor='0.8', ax=ax, color=df["color"].values)
        ax.legend(title=value_tag, handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

    edgecolor = "0.6"
    for shape, lw, _ in boundary_list:
        shape.plot(facecolor="none", linewidth=lw, edgecolor=edgecolor, ax=ax)
        edgecolor = "0.4"

    if print_labels:
        _add_labels_on_plot(df, color_column, ax, label_format_tot, labels_size=labels_size)

    if fig is not None:
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        if show_plot:
            plt.show()
    # return fig
    return fig, ax


def _add_labels_on_plot(df, value_column, ax, label_format, labels_size):
    labels_size = labels_size or 'large'
    if is_numeric_dtype(df[value_column]):
        total = df[value_column].sum()
        max_value = df[value_column].max()
        min_value = df[value_column].min()
        df["tx_color"] = np.where(df[value_column] > ((max_value + min_value) / 2), "white", "black")
        if cfg.TAG_REGIONE in df.columns:
            df.loc[df[cfg.TAG_REGIONE].isin(["Calabria", "Liguria"]), "tx_color"] = "black"
        for idx, row in df.iterrows():
            if row[value_column] > 0:
                text = label_format(row[value_column])
                ax.annotate(text, xy=(row['center_x'], row['center_y']), ha='center', va="center", fontsize=labels_size,
                            color=row["tx_color"], wrap=True)
    else:
        for idx, row in df.iterrows():
            ax.annotate(text=row[value_column],
                        xy=(row['center_x'], row['center_y']),
                        ha='center', va="center",
                        color='black', wrap=True, fontsize=labels_size)


simplify_values = {GeoLevel.REGIONE: 500,
                   GeoLevel.PROVINCIA: 500,
                   GeoLevel.COMUNE: 250}


def _plot_bokeh_choropleth_map(
    df: gpd.GeoDataFrame,
    level: GeoLevel,
    dict_values: Dict[str, str],
    shape_list: List,
    title: Optional[str] = None,
    title_size: str = "25px",
    fig_size: tuple = (800, 900),
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Plots a choropleth map with the specified parameters using Bokeh.

    Args:
        df (gpd.GeoDataFrame): DataFrame containing the data to plot.
        level (GeoLevel): Geographic level to plot (e.g., REGIONE, PROVINCIA, COMUNE).
        dict_values (Dict[str, str]): Dictionary of columns to plot.
        shape_list (List): List of shapes to be plotted on the map.
        title (str): Title of the plot. Defaults to None.
        title_size (str): Font size for the title. Defaults to "25px".
        fig_size (tuple): Size of the figure. Defaults to (800, 900).
        save_path (str): Path to save the plot. Defaults to None.
        show_plot (bool): Whether to display the plot. Defaults to True.
    """
    inverted_dict = {value: key for (key, value) in dict_values.items()}
    field_list = list(dict_values.keys())

    palette_list_numerical = [Blues9,
                              Greens9,
                              Reds9,
                              Purples9,
                              Oranges9,
                              Greys9]
    palette_list = {}
    legend_list = {}
    is_numeric = {}
    i = 0
    for key, value in dict_values.items():
        if is_numeric_dtype(df[key]):
            palette_list[key] = {"field": "values_plot", "transform": LinearColorMapper(
                palette=palette_list_numerical[i % len(palette_list_numerical)][::-1])}
            legend_list[key] = 0
            is_numeric[key] = True
            i += 1
        elif is_string_dtype(df[key]):
            values = list(df[key].unique())
            n_values = len(values)
            palette_list[key] = {"field": "values_plot", "transform": CategoricalColorMapper(factors=values,
                                                                                             palette=Category20[
                                                                                                 20] if n_values > 10 else
                                                                                             Category10[10])}
            is_numeric[key] = False
            legend_list[key] = 1
    df["values_plot"] = df[field_list[0]]
    df["line_color"] = "gray"
    df["geometry"] = df["geometry"].simplify(simplify_values[level])
    geosource = GeoJSONDataSource(geojson=df.to_json())
    geosource2 = ColumnDataSource(data=df[[get_tag_registry(CodeLevel.DENOMINATION, level)] + list(dict_values.keys())])
    mapper = palette_list[field_list[0]]
    p = figure(title=title,
               height=fig_size[1],
               width=fig_size[0],
               tools='pan, wheel_zoom, box_zoom, reset')
    if title is not None:
        p.title.text_font_size = title_size
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
    if level == GeoLevel.COMUNE:
        line_width = 0.1
        columns = [TableColumn(field=cfg.TAG_COMUNE, title=str(level).capitalize())] + columns
    elif level == GeoLevel.PROVINCIA:
        line_width = 0.25
        columns = [TableColumn(field=cfg.TAG_PROVINCIA, title=str(level).capitalize())] + columns
    elif level == GeoLevel.REGIONE:
        line_width = 0.5
        columns = [TableColumn(field=cfg.TAG_REGIONE, title=str(level).capitalize())] + columns
    else:
        line_width = 0.1

    data_table = DataTable(source=geosource2, columns=columns, selectable=False)
    image = p.patches('xs', 'ys', source=geosource,
                      fill_color=mapper,
                      fill_alpha=0.7,
                      # line_color='gray',
                      line_color='line_color',
                      line_width=line_width)
    line_color = "darkgray"
    for shape, lw, sf in shape_list:
        shape["geometry"] = shape["geometry"].simplify(simplify_values[sf])
        shape = GeoJSONDataSource(geojson=shape.to_json())
        p.patches('xs', 'ys', source=shape,
                  fill_alpha=0,
                  line_color=line_color,
                  line_width=lw)
        line_color = "black"

    tool_list = [(str(level).capitalize(), '@' + get_tag_registry(CodeLevel.DENOMINATION, level))]
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
        data = {'x': [df["geometry"].values[0].centroid.coords[0][0] for f in factors],
                'y': [df["geometry"].values[0].centroid.coords[0][1] for f in factors],
                'color': colors,
                'factor': factors}
        source_legend = ColumnDataSource(data=data)

    else:
        source_legend = ColumnDataSource(data={"x": [],
                                               "y": [],
                                               "color": [],
                                               "factor": []})
    legend_image = p.circle(x="x", y="y", size=0, fill_color="color", legend_field="factor", line_width=0,
                            source=source_legend)
    p.legend.title = list(dict_values.values())[0]
    p.legend.title_text_font_size = "20px"
    p.legend.title_text_font_style = "bold"
    if legend_list[field_list[0]] == 0:
        p.legend.border_line_width = 0

    geosource.selected.js_on_change('indices', CustomJS(
        args=dict(source=geosource, source2=geosource2),
        code="""
            var f = cb_obj.indices[0];
            console.log(f);
            var data = source.data;
            data["line_color"][f] = "blue";
            source.change.emit();
            var data2 = source2.data;
            data2["line_color"][f] = "blue";
            source2.change.emit();
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

    if save_path:
        output_file(save_path, mode='inline')
        save(plot)
    if show_plot:
        show(plot)

    return plot


@validate
def plot_point_map(
    df0: pd.DataFrame,
    latitude_column: Optional[str] = None,
    longitude_column: Optional[str] = None,
    geometry_column: Optional[str] = None,
    filter_comune: Union[str, List[str]] = None,
    filter_provincia: Union[str, List[str]] = None,
    filter_regione: Union[str, List[str]] = None,
    color_column: Optional[str] = None,
    color: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
    title_size: float = 20,
    legend_font: Union[int, float] = None,
    show_colorbar: bool = True,
    marker: str = "o",
    marker_size: int = 6,
    marker_alpha: float = 1,
    save_path: Union[str, Path] = None,
    add_map_background: bool = True,
    show_plot: bool = True,
    dpi: int = 100
) -> Axes:
    """
    Plots a point map based on the provided DataFrame and filters.

    Args:
        df0 (pd.DataFrame): DataFrame containing the data to plot.
        latitude_column (str, optional): Column name for latitude. Defaults to None.
        longitude_column (str, optional): Column name for longitude. Defaults to None.
        geometry_column (str, optional): Column name for geographic data. Defaults to None.
        filter_comune (Union[str, List[str]], optional): Filter for specific Comune. Defaults to None.
        filter_provincia (Union[str, List[str]], optional): Filter for specific Provincia. Defaults to None.
        filter_regione (Union[str, List[str]], optional): Filter for specific Regione. Defaults to None.
        color_column (str, optional): Column name for color coding. Defaults to None.
        color (str, optional): Color to use for the points. Defaults to None.
        ax (Axes, optional): Matplotlib Axes object to plot on. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        title_size (float, optional): Font size for the title. Defaults to 20.
        legend_font (Union[int, float], optional): Font size for the legend. Defaults to None.
        show_colorbar (bool, optional): Whether to display the colorbar. Defaults to True.
        marker (str, optional): Marker style. Defaults to "o".
        marker_size (int, optional): Size of the points. Defaults to 6.
        marker_alpha (float, optional): Transparency of the points. Defaults to 0.7.
        save_path (Union[str, Path], optional): Path to save the plot. Defaults to None.
        add_map_background (bool, optional): Whether to add a map background. Defaults to True.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        dpi (int, optional): Resolution of the saved plot. Defaults to 100.

    Returns:
        Axes: Matplotlib Axes object with the plot.
    """
    df = df0.copy()
    df = _create_geo_dataframe(df, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)
    df.to_crs(cfg.OPENSTREETMAP_CRS, inplace=True)

    filter_level, filter_list = _get_filter_params(filter_regione, filter_provincia, filter_comune)
    filter_list = ensure_list(filter_list)
    if filter_list is not None:
        shape = get_df(filter_level)
        shape = _filter_data(shape, filter_list, filter_level)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': cfg.SHAPE_CRS}
        shape.to_crs(cfg.OPENSTREETMAP_CRS, inplace=True)
        df = gpd.tools.sjoin(df, shape[["geometry"]], op='within')
        shape_list = [(shape, 0.4, "0.6")]
        if filter_level == GeoLevel.REGIONE:
            shape_list.extend(_get_additional_shapes(GeoLevel.PROVINCIA, filter_list, filter_level))
        elif filter_level == GeoLevel.PROVINCIA:
            shape_list.extend(_get_additional_shapes(GeoLevel.COMUNE, filter_list, filter_level))
    else:
        shape_list = _get_default_shapes()

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    if title:
        ax.set_title(title, fontsize=title_size)

    _plot_points(df, ax, color_column, color, marker, marker_size, marker_alpha, show_colorbar, legend_font)
    if add_map_background:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    _plot_shapes(ax, shape_list)

    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    if show_plot:
        plt.show()

    return ax


def _get_additional_shapes(level, filter_list, filter_level):
    shape = get_df(level)
    shape = _filter_data(shape, filter_list, filter_level)
    shape = gpd.GeoDataFrame(shape, geometry="geometry")
    shape.crs = {'init': cfg.SHAPE_CRS}
    shape.to_crs(cfg.OPENSTREETMAP_CRS, inplace=True)
    return [(shape, 0.2, "0.8")]


def _get_default_shapes():
    shapes = []
    for level, lw, ec in [(GeoLevel.REGIONE, 0.4, "0.6"), (GeoLevel.PROVINCIA, 0.2, "0.8")]:
        shape = get_df(level)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': cfg.SHAPE_CRS}
        shape.to_crs(cfg.OPENSTREETMAP_CRS, inplace=True)
        shapes.append((shape, lw, ec))
    return shapes


def _plot_points(df, ax, color_tag, color, marker, size, marker_alpha, show_colorbar, legend_font):
    if color_tag:
        if is_numeric_dtype(df[color_tag]):
            _plot_numeric_points(df, ax, color_tag, color, marker, size, marker_alpha, show_colorbar, legend_font)
        elif is_string_dtype(df[color_tag]):
            _plot_categorical_points(df, ax, color_tag, color, marker, size, marker_alpha, legend_font)
    else:
        ax.scatter(df.geometry.x, df.geometry.y, c=color, alpha=0.5, s=size)


def _plot_numeric_points(df, ax, color_tag, color, marker, size, marker_alpha, show_colorbar, legend_font):
    vmin, vmax = df[color_tag].min(), df[color_tag].max()
    cmap = get_cmap(color or "Blues")
    scatter = ax.scatter(df.geometry.x, df.geometry.y, c=df[color_tag], cmap=cmap, vmin=vmin, vmax=vmax,
                         alpha=marker_alpha,
                         linewidths=0.1, marker=marker, s=size, edgecolors=color or "blue")
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.ax.tick_params(labelsize=legend_font or 12)


def _plot_categorical_points(df, ax, color_tag, color, marker, size, marker_alpha, legend_font):
    color_labels = df[color_tag].unique()
    n_color = len(color_labels)
    color = color or ("tab10" if n_color <= 10 else "tab20")
    cmap = get_cmap(color)
    rgb_values = [cmap(i) for i in range(n_color)]
    color_map = dict(zip(color_labels, rgb_values))

    for label in color_labels:
        df_plot = df[df[color_tag] == label]
        ax.scatter(df_plot.geometry.x, df_plot.geometry.y, color=color_map[label], label=label, alpha=marker_alpha,
                   linewidths=0.5, s=size, marker=marker)

    ax.legend(loc="center left", title=color_tag, prop={'size': legend_font or 12},
              title_fontsize=(legend_font or 12) * 1.1, bbox_to_anchor=(1, 0.5))


def _plot_shapes(ax, shape_list):
    for shape, lw, ec in shape_list:
        shape.plot(facecolor="none", linewidth=lw, edgecolor=ec, ax=ax)


@validate("return", exclude=True)
def plot_point_map_interactive(
    df0: pd.DataFrame,
    latitude_column: str = None,
    longitude_column: str = None,
    geometry_column: str = None,
    filter_comune: Union[str, List[str]] = None,
    filter_provincia: Union[str, List[str]] = None,
    filter_regione: Union[str, List[str]] = None,
    color_column: str = None,
    info_dict: Dict[str, str] = None,
    title: str = None,
    table: bool = True,
    width: int = 1500,
    height: int = 800,
    save_path: Union[str, Path] = None,
    show_plot: bool = True
) -> Plot:
    """
    Plots an interactive point map based on the provided DataFrame and filters.

    Parameters:
    - df0 (pd.DataFrame): DataFrame containing the data to plot.
    - latitude_column (str, optional): Column name for latitude. Defaults to None.
    - longitude_column (str, optional): Column name for longitude. Defaults to None.
    - geometry_column (str, optional): Column name for geographic data. Defaults to None.
    - filter_comune (Union[str, List[str]], optional): Filter for specific Comune. Defaults to None.
    - filter_provincia (Union[str, List[str]], optional): Filter for specific Provincia. Defaults to None.
    - filter_regione (Union[str, List[str]], optional): Filter for specific Regione. Defaults to None.
    - color_tag (str, optional): Column name for color coding. Defaults to None.
    - info_dict (Dict[str, str], optional): Dictionary for additional information to display. Defaults to None.
    - title (str, optional): Title of the plot. Defaults to None.
    - table (bool, optional): Whether to display a data table. Defaults to True.
    - width (int, optional): Width of the plot. Defaults to 1500.
    - height (int, optional): Height of the plot. Defaults to 800.
    - save_path (Union[str, Path], optional): Path to save the plot. Defaults to None.
    - show_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
    - Plot: Bokeh Plot object.
    """
    margins, shape = _get_margins(filter_comune=filter_comune,
                                  filter_provincia=filter_provincia,
                                  filter_regione=filter_regione)

    plot = figure(x_range=(margins[0][0], margins[0][1]),
                  y_range=(margins[1][0], margins[1][1]),
                  x_axis_type="mercator", y_axis_type="mercator", width=width, height=height)
    plot.add_tile(xyz.CartoDB.Positron)

    if title is not None:
        plot.title.text = title
        plot.title.align = 'center'

    column_list = list(df0.columns)

    df = _create_geo_dataframe(df0, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)
    if latitude_column is None:
        latitude_column = "geo_ita_lat"
        longitude_column = "geo_ita_lon"
    df[latitude_column] = df.geometry.y
    df[longitude_column] = df.geometry.x
    df = df.to_crs({'init': cfg.OPENSTREETMAP_CRS})
    if filter_regione or filter_comune or filter_provincia:
        df = gpd.tools.sjoin(df, shape, op='within')

    if info_dict is not None:
        table_columns = list(info_dict.keys())
    else:
        table_columns = column_list[:10]
        table_columns += [color_column] if color_column is not None else []
        if "geometry" in table_columns:
            table_columns.remove("geometry")
    if (latitude_column is not None) and (latitude_column not in table_columns):
        table_columns.append(latitude_column)
        table_columns.append(longitude_column)

    table_columns = list(set(table_columns))

    df['x'] = df.geometry.x
    df['y'] = df.geometry.y
    df = pd.DataFrame(df.drop(columns='geometry'))

    source = ColumnDataSource(df)
    if info_dict is not None:
        columns = [TableColumn(field=a, title=b) for a, b in info_dict.items()]
    else:
        columns = [TableColumn(field=a, title=a) for a in table_columns if
                   a not in [longitude_column, latitude_column, geometry_column]]

    legend = False
    if color_column is not None:
        if is_numeric_dtype(df[color_column]):
            min_values = df[color_column].min()
            max_values = df[color_column].max()
            if min_values >= 0:
                exp_cmap = LinearColorMapper(palette=Reds9[::-1], low=0,
                                             high=max_values)
            else:
                palette_max = max(np.abs(min_values), max_values)
                exp_cmap = LinearColorMapper(palette=RdYlGn11[::-1], low=-palette_max,
                                             high=palette_max)
            fill_color = {'field': color_column, 'transform': exp_cmap}
        elif is_string_dtype(df[color_column]):
            values = list(df[color_column].unique())
            n_values = len(values)
            fill_color = {"field": color_column, "transform": CategoricalColorMapper(factors=values,
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
                            legend_field=color_column,
                            source=source)
        plot.legend.title = color_column
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
            if values not in [longitude_column, latitude_column, geometry_column]:
                if is_numeric_dtype(df[values]):
                    tooltips1.append((values, '@' + values + '{0.[0] a}'))
                else:
                    tooltips1.append((values, '@' + values))
    if (latitude_column is not None) and (longitude_column is not None):
        tooltips1.append(
            ("Coords", "(@" + latitude_column + "{0,0.[0000000]}-@" + longitude_column + "{0,0.[0000000]})"))

    plot.add_tools(HoverTool(renderers=[plot1], tooltips=tooltips1))

    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
    if table:
        data_table = DataTable(source=source, columns=columns)
        p = row(plot, data_table)
    else:
        p = plot
    if save_path is not None:
        output_file(save_path, mode='inline')
        save(p)
    if show_plot:
        if save_path is None:
            show(p)
        else:
            os.startfile(save_path)
    return p


@validate
def plot_density_map(
    df0: pd.DataFrame,
    latitude_column: Optional[str] = None,
    longitude_column: Optional[str] = None,
    geometry_column: Optional[str] = None,
    filter_comune: Union[str, List[str]] = None,
    filter_provincia: Union[str, List[str]] = None,
    filter_regione: Union[str, List[str]] = None,
    color_column: Optional[str] = None,
    color: Optional[str] = None,
    bw_method=None,
    log_scale: bool = False,
    ax=None,
    title: Optional[str] = None,
    title_size: float = 20,
    save_path: Union[str, Path] = None,
    add_map_background: bool = True,
    show_plot: bool = True,
    dpi: int = 100
) -> Axes:
    """
    Plots a density map based on the provided DataFrame and filters.

    Args:
        df0 (pd.DataFrame): DataFrame containing the data to plot.
        latitude_column (str, optional): Column name for latitude. Defaults to None.
        longitude_column (str, optional): Column name for longitude. Defaults to None.
        geometry_column (str, optional): Column name for geographic data. Defaults to None.
        filter_comune (Union[str, List[str]], optional): Filter for specific Comune. Defaults to None.
        filter_provincia (Union[str, List[str]], optional): Filter for specific Provincia. Defaults to None.
        filter_regione (Union[str, List[str]], optional): Filter for specific Regione. Defaults to None.
        color_column (str, optional): Column name for color coding. Defaults to None.
        color (str, optional): Color to use for the points. Defaults to None.
        bw_method (float, optional): Bandwidth method for the KDE. Defaults to None.
        log_scale (bool, optional): Whether to use a logarithmic scale. Defaults to False.
        ax (Axes, optional): Matplotlib Axes object to plot on. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        title_size (float, optional): Font size for the title. Defaults to 20.
        save_path (Union[str, Path], optional): Path to save the plot. Defaults to None.
        add_map_background (bool, optional): Whether to add a map background. Defaults to True.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        dpi (int, optional): Resolution of the saved plot. Defaults to 100.

    Returns:
        Axes: Matplotlib Axes object with the plot.
    """
    df = df0.copy()
    df = _create_geo_dataframe(df, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)
    df.to_crs(cfg.OPENSTREETMAP_CRS, inplace=True)

    filter_level, filter_list = _get_filter_params(filter_regione, filter_provincia, filter_comune)
    filter_list = ensure_list(filter_list)
    if filter_list is not None:
        shape = get_df(filter_level)
        shape = _filter_data(shape, filter_list, filter_level)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")
        shape.crs = {'init': cfg.SHAPE_CRS}
        shape.to_crs(cfg.OPENSTREETMAP_CRS, inplace=True)
        df = gpd.tools.sjoin(df, shape[["geometry"]], op='within')
        shape_list = [(shape, 0.4, "0.6")]
        if filter_level == GeoLevel.REGIONE:
            shape_list.extend(_get_additional_shapes(GeoLevel.PROVINCIA, filter_list, filter_level))
        elif filter_level == GeoLevel.PROVINCIA:
            shape_list.extend(_get_additional_shapes(GeoLevel.COMUNE, filter_list, filter_level))
    else:
        shape_list = _get_default_shapes()
        shape = get_df(GeoLevel.REGIONE)
        shape = gpd.GeoDataFrame(shape, geometry="geometry")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': gcrs.WebMercator()})

    if title:
        ax.set_title(title, fontsize=title_size)

    _plot_density(df, ax, color_column, color, bw_method, log_scale, shape)
    if add_map_background:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    _plot_shapes(ax, shape_list)

    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    if show_plot:
        plt.show()

    return ax


def _plot_density(df, ax, color_tag, color, bw_method, log_scale, shape):
    shape["geometry"] = shape["geometry"].buffer(0)
    shape = shape.explode(index_parts=False)
    shape = shape[~shape["geometry"].is_empty]

    df.to_crs(cfg.GEOPLOT_CRS, inplace=True)
    shape.to_crs(cfg.GEOPLOT_CRS, inplace=True)
    df = df.dropna()
    weights = df[color_tag].values if color_tag else None
    gplt.kdeplot(
        df, projection=gcrs.WebMercator(),
        weights=weights, color=color or "Blue", cbar=False,
        bw_method=bw_method, fill=True, log_scale=log_scale, thresh=0.05, alpha=0.7,
        clip=shape[["geometry"]],
        ax=ax,
    )
