import logging
import os
from pathlib import Path
from typing import Union, Optional

from bokeh.models import (
    ColumnDataSource, DataTable, TableColumn, HTMLTemplateFormatter, CategoricalColorMapper,
    LabelSet, Label, WheelZoomTool, CustomJS, TabPanel, Tabs, TextInput, HoverTool
)
from bokeh.plotting import save, figure
from bokeh.io import output_file, show
from bokeh.layouts import column, row
from valdec.decorators import validate
import pandas as pd
import numpy as np
from shapely import Point
import xyzservices.providers as xyz

from geo_ita.src._data_enrichment import AddGeographicalInfo, get_city_from_coordinates
from geo_ita.src.utils import GeoLevel, CodeLevel, infer_geographical_category, get_tag_registry, test_column_in_dataframe
import geo_ita.src.config as cfg

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pyproj import Proj, transform
inProj, outProj = Proj(init='epsg:4326'), Proj(init='epsg:3857')


class GeoDataQuality:
    POINT_COLUMN = "geometry"
    CHECK_SUFFIX = "_check"
    CORRECTION_SUFFIX = "_correction"
    FLAG_ITALY_COLUMN = "is_in_italy"
    DATA_QUALITY_CHECK_TAG = "check_data_quality_add_info"

    @validate
    def __init__(self, df: pd.DataFrame):
        self.original_df = df
        self.unique_key_column = None
        self.italy_name = "italy"
        self.detail_level = {}

    @validate
    def set_unique_key_column(self, column_name: str):
        test_column_in_dataframe(self.original_df, column_name)
        if not self.original_df[column_name].is_unique:
            raise Exception("Insert a column with unique values.")
        self.unique_key_column = column_name

    @validate
    def set_comuni_tag(self, column_name: str):
        test_column_in_dataframe(self.original_df, column_name)
        code_level = infer_geographical_category(list(self.original_df[column_name].unique()))
        if code_level == CodeLevel.SIGLA:
            raise Exception(f"Found values in {column_name} similar to Province Sigla. "
                            f"Check the column name passed and the values on columns.")
        self.detail_level[GeoLevel.COMUNE] = (column_name, code_level)

    @validate
    def set_province_tag(self, column_name: str):
        test_column_in_dataframe(self.original_df, column_name)
        code_level = infer_geographical_category(list(self.original_df[column_name].unique()))
        self.detail_level[GeoLevel.PROVINCIA] = (column_name, code_level)

    @validate
    def set_regioni_tag(self, column_name: str):
        test_column_in_dataframe(self.original_df, column_name)
        code_level = infer_geographical_category(list(self.original_df[column_name].unique()))
        if code_level == CodeLevel.SIGLA:
            raise Exception(f"Found values in {column_name} similar to Province Sigla. "
                            f"Check the column name passed and the values on columns.")
        self.detail_level[GeoLevel.REGIONE] = (column_name, code_level)

    @validate
    def set_country_tag(self, column_name: str):
        test_column_in_dataframe(self.original_df, column_name)
        self.detail_level[GeoLevel.COUNTRY] = (column_name, CodeLevel.DENOMINATION)

    @validate
    def set_latitude_longitude_tag(
        self,
        longitude_column: Optional[str] = None,
        latitude_column: Optional[str] = None,
        geometry_column: Optional[str] = None,
    ):
        if longitude_column is not None:
            test_column_in_dataframe(self.original_df, latitude_column)
            test_column_in_dataframe(self.original_df, longitude_column)
        else:
            test_column_in_dataframe(self.original_df, geometry_column)

        # Create point column
        if longitude_column is not None:
            self.original_df[self.POINT_COLUMN + "_x"] = pd.to_numeric(self.original_df[longitude_column], errors="coerce")
            self.original_df[self.POINT_COLUMN + "_y"] = pd.to_numeric(self.original_df[latitude_column], errors="coerce")
            self.original_df[self.POINT_COLUMN] = self.original_df.apply(
                lambda row: Point(row[self.POINT_COLUMN + "_x"], row[self.POINT_COLUMN + "_y"]),
                axis=1)
            self.original_df.drop(columns=[self.POINT_COLUMN + "_x", self.POINT_COLUMN + "_y"], inplace=True)
        else:
            self.original_df[self.POINT_COLUMN] = self.original_df[geometry_column]
        self.detail_level[GeoLevel.COORDINATES] = (self.POINT_COLUMN, CodeLevel.COORDINATES)

    def _first_check(self, geo_level):
        original_column = self.detail_level[geo_level][0]
        original_code = self.detail_level[geo_level][1]
        tag = get_tag_registry(CodeLevel.DENOMINATION, geo_level)

        # Clean column
        if original_code == CodeLevel.DENOMINATION:
            self.original_df[tag] = self._clean_denomination(self.original_df[original_column])
        elif original_code == CodeLevel.CODE:
            self.original_df[tag] = self.original_df[original_column].astype("Int", errors="coerce")
        else:
            self.original_df[tag] = self.original_df[original_column]

        # Check for missing values
        self.original_df[tag + self.CHECK_SUFFIX] = self.original_df[original_column].isna()

    def _second_check(self, geo_level):
        original_code = self.detail_level[geo_level][1]
        tag = get_tag_registry(CodeLevel.DENOMINATION, geo_level)

        self.original_df[tag + self.CHECK_SUFFIX] |= self.original_df[tag + "_" + geo_level].isna()
        if GeoLevel.COUNTRY in self.detail_level:
            country_column = self.detail_level[GeoLevel.COUNTRY][0]
            self.original_df[get_tag_registry(CodeLevel.DENOMINATION, GeoLevel.COUNTRY) + "_" + geo_level] = np.where(
                self.original_df[tag + "_" + geo_level].notnull(),
                self.italy_name,
                self.original_df[country_column]
            )

        # Check original value
        self.original_df[tag + self.CORRECTION_SUFFIX] = None
        if original_code == CodeLevel.DENOMINATION:
            wrong_position = (self.original_df[tag + "_" + geo_level] != self.original_df[tag]) & \
                             self.original_df[tag + "_" + geo_level].notnull()
            self.original_df.loc[wrong_position, tag + self.CHECK_SUFFIX] = True
            self.original_df.loc[wrong_position, tag + self.CORRECTION_SUFFIX] = self.original_df.loc[
                wrong_position, tag + "_" + geo_level]

    def _check_two_level(self, level1, level2):
        tag = get_tag_registry(CodeLevel.DENOMINATION, level1)
        check_pos = (self.original_df[tag + "_" + level2].notnull() &
                     (self.original_df[tag + "_" + level2] != self.original_df[tag + "_" + level1]))
        self.original_df.loc[check_pos, tag + self.CHECK_SUFFIX] = True
        fill_na_pos = check_pos & self.original_df[tag + "_" + level1].isna() & self.original_df[
            tag + self.CORRECTION_SUFFIX].isna()
        self.original_df.loc[fill_na_pos, tag + self.CORRECTION_SUFFIX] = self.original_df.loc[
            fill_na_pos, tag + "_" + level2]
        different_correction_pos = (
                check_pos &
                self.original_df[tag + "_" + level1].isna() &
                (self.original_df[tag + self.CORRECTION_SUFFIX] != self.original_df[tag + "_" + level2])
        )
        self.original_df.loc[different_correction_pos, tag + self.CORRECTION_SUFFIX] = None

    def _check_country(self):
        self._first_check(GeoLevel.COUNTRY)

        original_column = self.detail_level[GeoLevel.COUNTRY][0]
        tag = get_tag_registry(CodeLevel.DENOMINATION, GeoLevel.COUNTRY)

        italy_string_names = ["it", "italy", "italia", "ita"]
        # Get italy name
        values = self.original_df.loc[
            self.original_df[original_column].isin(italy_string_names), original_column].value_counts()
        if values.shape[0] >= 1:
            self.italy_name = values.index[0]

        self.original_df[tag + "_" + GeoLevel.COUNTRY] = self.original_df[original_column].map(
            {name: self.italy_name for name in italy_string_names})

        self._second_check(GeoLevel.COUNTRY)

    @staticmethod
    def _clean_denomination(series):
        series = series.str.lower()  # All strig in lowercase
        series = series.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non alphabetic characters
        series = series.str.strip()
        series = series.str.replace(r'\s+', ' ', regex=True)
        series = series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode(
            'utf-8')  # Remove accent
        series = series.replace("", None)
        return series

    def _check_regione(self):
        self._first_check(GeoLevel.REGIONE)

        regione_column = self.detail_level[GeoLevel.REGIONE][0]
        regione_code = self.detail_level[GeoLevel.REGIONE][1]

        # Start finding the right name
        addinfo = AddGeographicalInfo(self.original_df)
        addinfo.set_regioni_tag(regione_column)
        addinfo.run_simple_match()
        if regione_code == CodeLevel.DENOMINATION:
            addinfo.run_similarity_match(threshold=0.85)
            addinfo.accept_similarity_result()
        check_df = addinfo.get_result(suffix_result_columns=self.DATA_QUALITY_CHECK_TAG)
        self.original_df[cfg.TAG_REGIONE + "_" + GeoLevel.REGIONE] = self._clean_denomination(
            check_df[cfg.TAG_REGIONE + self.DATA_QUALITY_CHECK_TAG])

        self._second_check(GeoLevel.REGIONE)

    def _check_provincia(self):
        self._first_check(GeoLevel.PROVINCIA)

        provincia_column = self.detail_level[GeoLevel.PROVINCIA][0]
        provincia_code = self.detail_level[GeoLevel.PROVINCIA][1]

        # Start finding the right name
        addinfo = AddGeographicalInfo(self.original_df)
        addinfo.set_province_tag(provincia_column)
        addinfo.run_simple_match()
        if provincia_code == CodeLevel.DENOMINATION:
            addinfo.run_similarity_match(threshold=0.85)
            addinfo.accept_similarity_result()
        check_df = addinfo.get_result(suffix_result_columns=self.DATA_QUALITY_CHECK_TAG)
        self.original_df[cfg.TAG_REGIONE + "_" + GeoLevel.PROVINCIA] = self._clean_denomination(
            check_df[cfg.TAG_REGIONE + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_PROVINCIA + "_" + GeoLevel.PROVINCIA] = self._clean_denomination(
            check_df[cfg.TAG_PROVINCIA + self.DATA_QUALITY_CHECK_TAG])

        self._second_check(GeoLevel.PROVINCIA)

    def _check_comune(self):
        self._first_check(GeoLevel.COMUNE)

        comune_column = self.detail_level[GeoLevel.COMUNE][0]
        comune_code = self.detail_level[GeoLevel.COMUNE][1]

        # Start finding the right name
        addinfo = AddGeographicalInfo(self.original_df)
        addinfo.set_comuni_tag(comune_column)
        if GeoLevel.PROVINCIA in self.detail_level:
            addinfo.set_province_tag(self.detail_level[GeoLevel.PROVINCIA][0])
        if GeoLevel.REGIONE in self.detail_level:
            addinfo.set_regioni_tag(self.detail_level[GeoLevel.REGIONE][0])
        addinfo.run_simple_match()
        if comune_code == CodeLevel.DENOMINATION:
            addinfo.run_find_frazioni()
            addinfo.run_find_frazioni_on_web()
            addinfo.run_similarity_match(threshold=0.85)
            addinfo.accept_similarity_result()
        check_df = addinfo.get_result(suffix_result_columns=self.DATA_QUALITY_CHECK_TAG)
        self.original_df[cfg.TAG_REGIONE + "_" + GeoLevel.COMUNE] = self._clean_denomination(
            check_df[cfg.TAG_REGIONE + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_PROVINCIA + "_" + GeoLevel.COMUNE] = self._clean_denomination(
            check_df[cfg.TAG_PROVINCIA + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_COMUNE + "_" + GeoLevel.COMUNE] = self._clean_denomination(
            check_df[cfg.TAG_COMUNE + self.DATA_QUALITY_CHECK_TAG])

        self._second_check(GeoLevel.COMUNE)

    def _check_coordinates(self):
        self._first_check(GeoLevel.COORDINATES)

        check_df = get_city_from_coordinates(self.original_df, suffix_result_columns=self.DATA_QUALITY_CHECK_TAG)
        self.original_df[cfg.TAG_REGIONE + "_" + GeoLevel.COORDINATES] = self._clean_denomination(
            check_df[cfg.TAG_REGIONE + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_PROVINCIA + "_" + GeoLevel.COORDINATES] = self._clean_denomination(
            check_df[cfg.TAG_PROVINCIA + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_COMUNE + "_" + GeoLevel.COORDINATES] = self._clean_denomination(
            check_df[cfg.TAG_COMUNE + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_COORDINATES + "_" + GeoLevel.COORDINATES] = self.original_df[
            cfg.TAG_COMUNE + "_" + GeoLevel.COORDINATES].where(
            self.original_df[cfg.TAG_COMUNE + "_" + GeoLevel.COORDINATES].notnull()
        )

        self._second_check(GeoLevel.COORDINATES)

    @validate
    def start_check(self) -> None:

        column_list = [v[0] for v in self.detail_level.values()]
        if self.unique_key_column is not None:
            column_list = [self.unique_key_column] + column_list
        self.original_df = self.original_df[column_list].drop_duplicates()

        # Create a flag for italy positions
        if GeoLevel.COUNTRY in self.detail_level:
            self._check_country()

        if GeoLevel.REGIONE in self.detail_level:
            self._check_regione()
            if GeoLevel.COUNTRY in self.detail_level:
                self._check_two_level(GeoLevel.COUNTRY, GeoLevel.REGIONE)

        if GeoLevel.PROVINCIA in self.detail_level:
            self._check_provincia()
            if GeoLevel.COUNTRY in self.detail_level:
                self._check_two_level(GeoLevel.COUNTRY, GeoLevel.PROVINCIA)
            if GeoLevel.REGIONE in self.detail_level:
                self._check_two_level(GeoLevel.REGIONE, GeoLevel.PROVINCIA)

        if GeoLevel.COMUNE in self.detail_level:
            self._check_comune()
            if GeoLevel.COUNTRY in self.detail_level:
                self._check_two_level(GeoLevel.COUNTRY, GeoLevel.COMUNE)
            if GeoLevel.REGIONE in self.detail_level:
                self._check_two_level(GeoLevel.REGIONE, GeoLevel.COMUNE)
            if GeoLevel.PROVINCIA in self.detail_level:
                self._check_two_level(GeoLevel.PROVINCIA, GeoLevel.COMUNE)

        if GeoLevel.COORDINATES in self.detail_level:
            self._check_coordinates()
            if GeoLevel.COUNTRY in self.detail_level:
                self._check_two_level(GeoLevel.COUNTRY, GeoLevel.COORDINATES)
            if GeoLevel.REGIONE in self.detail_level:
                self._check_two_level(GeoLevel.REGIONE, GeoLevel.COORDINATES)
            if GeoLevel.PROVINCIA in self.detail_level:
                self._check_two_level(GeoLevel.PROVINCIA, GeoLevel.COORDINATES)
            if GeoLevel.COMUNE in self.detail_level:
                self._check_two_level(GeoLevel.COMUNE, GeoLevel.COORDINATES)

        check_list = [col for col in self.original_df.columns if self.CHECK_SUFFIX in col]
        self.original_df["check"] = self.original_df[check_list].sum(axis='columns')
        solved_list = [col for col in self.original_df.columns if self.CORRECTION_SUFFIX in col]
        self.original_df["solved"] = (self.original_df[solved_list].notnull()).sum(axis='columns')
        self.original_df["solved"] = (self.original_df["solved"] > 0) & (
                    self.original_df["solved"] == self.original_df["check"])
        self.original_df["check"] = self.original_df["check"] > 0

        n_tot = self.original_df.shape[0]

        n_check = self.original_df["check"].sum()
        n_solved = self.original_df["solved"].sum()
        log.info(f"Found {n_check} problems over {n_tot} ({n_check / n_tot:.2%}), "
                 f"of which {n_solved} solved ({n_solved / n_check:.2%}).")

        for geo_level, v in self.detail_level.items():
            tag = get_tag_registry(CodeLevel.DENOMINATION, geo_level)
            original_column = v[0]
            n_problem = self.original_df[tag + self.CHECK_SUFFIX].sum()
            n_problem_solved = self.original_df[tag + self.CORRECTION_SUFFIX].notnull().sum()
            log.info(f"Column {original_column}: {n_problem} problem, {n_problem_solved} solved.")
        return

    def get_results(self) -> pd.DataFrame:
        return self.original_df[self.original_df["check"]]

    @staticmethod
    def _create_header(width, background_color, text_color, title, subtitle):
        height = 100 if subtitle is not None else 50
        header = figure(x_range=(0, 1), y_range=(0, 1),
                        width=width, height=height,
                        tools="")
        header.background_fill_color = background_color
        header.xgrid.grid_line_color = None
        header.ygrid.grid_line_color = None
        header.axis.visible = False
        header.toolbar.logo = None
        header.outline_line_color = None
        header.toolbar_location = None
        header.min_border_left = 0
        header.min_border_right = 0
        header.min_border_top = 0
        header.min_border_bottom = 0

        header.add_layout(
            Label(x=0.005, y=.8, text=title,
                  text_font_style="bold",
                  text_font_size="20pt",
                  text_baseline="top",
                  text_color=text_color))

        if subtitle is not None:
            header.add_layout(
                Label(x=0.005, y=.4, text=subtitle,
                      text_font_size="12pt",
                      text_baseline="top",
                      text_color=text_color))
        return header

    def _create_map_plot(self, width, source_info):

        margins = [[723576.6901562785, 2070542.52875489], [4355801.264971882, 5999391.278141545]]

        map_plot = figure(x_range=(margins[0][0], margins[0][1]),
                          y_range=(margins[1][0], margins[1][1]),
                          x_axis_type="mercator", y_axis_type="mercator", width=width,
                          tools='pan,tap,wheel_zoom')
        map_plot.add_tile(xyz.CartoDB.Positron)
        map_plot.xgrid.grid_line_color = None
        map_plot.ygrid.grid_line_color = None
        map_plot.yaxis.visible = False
        map_plot.grid.visible = False
        map_plot.toolbar.logo = None
        map_plot.outline_line_color = None
        map_plot.xaxis.major_tick_line_color = None
        map_plot.xaxis.minor_tick_line_color = None
        map_plot.xaxis.major_label_text_font_size = '0pt'

        plot1 = map_plot.circle(x="longitudine_marcator", y="latitudine_marcator",
                                size=7,
                                fill_alpha="selected_alpha",
                                line_color="gray", line_width=0.5, source=source_info)
        # plot1 = map_plot.add_glyph(source_info, points)

        tooltips = [(self.keys, "@" + self.keys)]
        if self.nazione_tag is not None:
            tooltips.append(("Nazione", "@" + self.nazione_tag))
        if self.regioni_tag is not None:
            tooltips.append(("Regione", "@" + self.regioni_tag))
        if self.province_tag is not None:
            tooltips.append(("Provincia", "@" + self.province_tag))
        if self.comuni_tag is not None:
            tooltips.append(("Comune", "@" + self.comuni_tag))

        tooltips.append(
            ("Coordinates", "(@" + self.latitude_tag + "{0,0.0000000}-@" + self.longitude_tag + "{0,0.0000000})"))

        map_plot.add_tools(HoverTool(renderers=[plot1], tooltips=tooltips))

        map_plot.toolbar.active_scroll = map_plot.select_one(WheelZoomTool)

        return map_plot

    def _create_text_key_copy(self):
        text_input = TextInput(value="", title=self.keys + ": ", width=200)
        return text_input

    @validate
    def plot_result(self, background_color: str = "white", text_color: str = "black",
                    title: str = "Geographical DataQuality",
                    subtitle: str = None,
                    show_only_warning: bool = True,
                    save_in_path: Union[str, Path] = None):
        n_tot = self.original_df.shape[0]
        width = 1000
        width_check = 250
        row_height = 30
        perc_height = 50
        ok_tag = "OK"
        warning_tag = "Warning"
        solved_tag = "Warning solved"
        if show_only_warning:
            source = self.original_df[self.original_df["check"]]
        else:
            source = self.original_df
        n_plot = source.shape[0]
        height = (n_plot + 1) * row_height
        source["x"] = 0.5
        source["y"] = range(n_plot)[::-1]
        source["y"] += 0.5
        source["check_color"] = ok_tag
        source["selected_color"] = "transparent"
        source["selected_alpha"] = 1
        if self.latitude_tag:
            source['longitudine_marcator'], source['latitudine_marcator'] = transform(inProj, outProj,
                                                                                      source[
                                                                                          self.longitude_tag].tolist(),
                                                                                      source[
                                                                                          self.latitude_tag].tolist())
        pos = source["check"]
        source.loc[pos, "check_color"] = warning_tag
        pos = source["solved"]
        source.loc[pos, "check_color"] = solved_tag

        if self.keys is None:
            self.keys = "index"
            source = source.reset_index().rename(columns={source.index.name: self.keys})

        source[self.keys] = source[self.keys].astype(str)

        # source = source.where(pd.notnull(source), None)
        propose_col = [a for a in source.columns if self.propose_tag in a]
        source[propose_col] = source[propose_col].fillna("")

        col_list = [self.nazione_tag, self.regioni_tag, self.province_tag, self.comuni_tag]
        col_list = [a for a in col_list if a is not None]

        template = """
               <div style="background:<%= 
               (function colorfromint(){{
                        return(selected_color)
                            }}()) %>; 
                   color: black">
               <%= value %> 
               </div>
               """
        formatter = HTMLTemplateFormatter(template=template)
        columns = [TableColumn(field=self.keys, title=self.keys, formatter=formatter)]

        tag_mapping = {self.nazione_tag: (None, None),
                       self.regioni_tag: (self.regioni_result_tag, "Regione"),
                       self.province_tag: (self.province_result_tag, "Provincia"),
                       self.comuni_tag: (self.comuni_result_tag, "Comune")}

        perc_data = []
        legend_data = []
        html_tag = "_html"
        i = 1
        for c in col_list:
            n_check = (source[c + self.check_tag] & (source[c + self.propose_tag] == "")).sum()
            perc_data.append([i + 0.8, 0.75, "{} ({}%)".format(n_check, int(round(n_check / n_tot * 100, 0)))])
            n_propose = (source[c + self.propose_tag] != "").sum()
            perc_data.append([i + 0.8, 0.25, "{} ({}%)".format(n_propose, int(round(n_propose / n_tot * 100, 0)))])
            legend_data.append([i + 0.9, 0.75, warning_tag, c + self.check_tag])
            legend_data.append([i + 0.9, 0.25, solved_tag, c + self.check_tag])
            pos = source[c + self.propose_tag] != ""
            # source[c].fillna("NaN", inplace=True)
            original = source[c].fillna("NaN").copy()
            source[c + html_tag] = np.where(pos,
                                            original + "||" + source[c + self.propose_tag],
                                            " ||" + original)
            source[c + html_tag] = source[c + html_tag] + "||" + original + "||" + source[c + self.propose_tag]
            tag, name = tag_mapping[c]
            html_tooltip = ""
            if tag is not None:
                if tag + "_regione" in source.columns:
                    html_tooltip += "\n{} found from regione: <%= value.split('||')[4] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_regione"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
                if tag + "_provincia" in source.columns:
                    html_tooltip += "\n{} found from provincia: <%= value.split('||')[5] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_provincia"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
                if tag + "_comune" in source.columns:
                    html_tooltip += "\n{} found from comune: <%= value.split('||')[6] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_comune"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
                if tag + "_coordinates" in source.columns:
                    html_tooltip += "\n{} found from coordinates: <%= value.split('||')[7] %>".format(name)
                    source[c + html_tag] += "||" + source[tag + "_coordinates"].fillna("-")
                else:
                    source[c + html_tag] += "|| "
            template = """
                        <div style="background:<%= 
                            (function colorfromint(){{
                                if({check}){{
                                    if({propose} != ""){{
                                        return("orange")
                                        }} else {{
                                        return("red")
                                    }}
                                }}
                            }}()) %>; 
                            color: black">
                        <span href="#" data-toggle="tooltip" title="Original: <%= value.split('||')[2] %>\nSuggestion: <%= value.split('||')[3] %>{html_tooltip}">
                            <strike><%=  value.split("||")[0] %></strike> <%= value.split("||")[1] %>
                        </span>
                        </div>
                        """.format(check=c + self.check_tag,
                                   propose=c + self.propose_tag,
                                   html_tooltip=html_tooltip)
            formatter = HTMLTemplateFormatter(template=template)
            columns.append(TableColumn(field=c + html_tag, title=c, formatter=formatter))
            i += 1

        if self.latitude_tag:
            n_check = source["coordinates" + self.check_tag].sum()
            perc_data.append([i + 0.8, 0.75, "{} ({}%)".format(n_check, int(round(n_check / n_tot * 100, 0)))])
            legend_data.append([i + 0.9, 0.75, warning_tag, "coordinates" + self.check_tag])
            i += 1
            perc_data.append([i + 0.8, 0.75, "{} ({}%)".format(n_check, int(round(n_check / n_tot * 100, 0)))])
            legend_data.append([i + 0.9, 0.75, warning_tag, "coordinates" + self.check_tag])
            i += 1
            template = """
                        <div style="background:<%= 
                            (function colorfromint(){{
                                if({}){{
                                    return("red")}}
                                }}()) %>; 
                            color: black"> 
                        <%= value %>
                        </div>
                        """.format("coordinates" + self.check_tag)
            formatter = HTMLTemplateFormatter(template=template)
            columns.append(
                TableColumn(field=self.latitude_tag, title=self.latitude_tag, formatter=formatter))
            columns.append(
                TableColumn(field=self.longitude_tag, title=self.longitude_tag, formatter=formatter))

        header = self._create_header(width + width_check, background_color, text_color, title, subtitle)
        text_input = self._create_text_key_copy()
        column_drop = ["is_in_italy", self.regioni_tag + "_regione", self.regioni_tag + "_provincia",
                       self.regioni_tag + "_comune", self.regioni_tag + "_coordinates",
                       self.province_tag + "_provincia", self.province_tag + "_comune",
                       self.province_tag + "_coordinates",
                       self.comuni_tag + "_comune", self.comuni_tag + "_coordinates"]
        column_drop = [a for a in column_drop if a in source.columns]
        source = source.drop(column_drop, axis=1)
        originalsource = ColumnDataSource(source)
        source = ColumnDataSource(source)
        data_table = DataTable(source=source,
                               columns=columns,
                               fit_columns=True,
                               selectable=True,
                               sortable=False,
                               editable=True,
                               index_position=None,
                               row_height=row_height,
                               height=height, width=width)

        source.selected.js_on_change('indices',
                                     CustomJS(args=dict(source=source,
                                                        text=text_input), code="""
                            var indices = cb_obj.indices;
                            console.log(cb_obj)
                            if (indices.length > 0){{

                                var current_value = text.value; 
                                var pos = cb_obj.indices[0];
                                console.log("Selected", pos) 
                                var data = source.data;

                                var selected_value = data["{key}"][pos];
                                if (current_value != selected_value){{
                                    for (var i = 0; i < data["{key}"].length; i++) {{
                                        data["selected_alpha"][i] = 0;
                                    }}
                                }} else {{
                                    for (var i = 0; i < data["{key}"].length; i++) {{
                                        data["selected_alpha"][i] = 1;
                                    }}
                                }}
                                if (current_value == "") {{
                                    text.value = selected_value;
                                    data["selected_color"][pos] = "yellow";
                                    data["selected_alpha"][pos] = 1;
                                }} else {{
                                    for (var i = 0; i < data["{key}"].length; i++) {{
                                        data["selected_color"][i] = "transparent";
                                    }}
                                    if (current_value != selected_value) {{
                                        text.value = selected_value;
                                        data["selected_color"][pos] = "yellow";
                                        data["selected_alpha"][pos] = 1;
                                    }} else {{
                                        text.value = "";
                                    }}
                                }}
                                source.change.emit();
                            }}
                            console.log("Ended")
                            cb_obj.indices = [];
                        """.format(key=self.keys))
                                     )

        check_plot = figure(
            height=height,
            width=width_check,
            x_range=(0, 1),
            y_range=(0, n_plot),
            x_axis_location="above",
            tools='')
        check_plot.xgrid.grid_line_color = None
        check_plot.ygrid.grid_line_color = None
        check_plot.yaxis.visible = False
        check_plot.grid.visible = False
        check_plot.toolbar.logo = None
        check_plot.outline_line_color = None
        check_plot.xaxis.major_label_text_font_size = '10pt'
        check_plot.xaxis.ticker = [0.5]
        check_plot.xaxis.major_label_overrides = {0.5: "Check"}

        check_plot.circle(x="x", y="y", size=9, line_width=0.5,
                          fill_color={"field": "check_color",
                                      "transform": CategoricalColorMapper(factors=[ok_tag, solved_tag, warning_tag],
                                                                          palette=["green", "orange", "red"])},
                          source=source, legend_label="check_color")
        check_plot.add_layout(check_plot.legend[0], 'right')

        perc_plot = figure(
            height=perc_height,
            width=width,
            x_range=(0, i),
            y_range=(0, 1),
            tools='tap')
        perc_plot.title.text_font_size = '16pt'
        perc_plot.xgrid.grid_line_color = None
        perc_plot.ygrid.grid_line_color = None
        perc_plot.yaxis.visible = False
        perc_plot.xaxis.visible = False
        perc_plot.grid.visible = False
        perc_plot.toolbar.logo = None
        perc_plot.outline_line_color = None
        perc_plot.xaxis.major_tick_line_color = None
        perc_plot.xaxis.minor_tick_line_color = None
        perc_plot.xaxis.major_label_text_font_size = '0pt'
        perc_plot.toolbar_location = None
        legend_data = np.array(legend_data)
        legend_data = ColumnDataSource(dict(
            x=legend_data[:, 0].astype(float),
            y=legend_data[:, 1].astype(float),
            color=legend_data[:, 2],
            column=legend_data[:, 3],
            alpha=np.ones(legend_data.shape[0]) * 0.5))

        legend_data.selected.js_on_change('indices',
                                          CustomJS(args=dict(source=source,
                                                             original_source=originalsource,
                                                             legend_source=legend_data), code="""
                                    var indices = cb_obj.indices;
                                    if (indices.length > 0){
                                        var df_legend = legend_source.data;
                                        var pos = cb_obj.indices[0];
                                        console.log("Selected", pos) 
                                        var data = source.data;
                                        var column_selected = df_legend["column"][pos];
                                        var color_selected = df_legend["y"][pos];
                                        var previus_selected = (df_legend["alpha"][pos] == 1);
                                        console.log("Previus selected", previus_selected) 
                                        var df0 = original_source.data;
                                        var df = source.data;
                                        if (previus_selected){
                                            df_legend["alpha"][pos] = 0.5
                                            for (var key in df0) {
                                                df[key] = [];
                                                for (var i = 0; i < df0[key].length; ++i) {
                                                    df[key].push(df0[key][i]);
                                                }
                                            }
                                        } else {
                                            df_legend["alpha"][pos] = 1
                                            for (var key in df0) {
                                                var y_val = df0[key].length + 0.5
                                                df[key] = [];
                                                for (i = 0; i < df0[key].length;i++){
                                                    if (column_selected.includes("coordinates")) {
                                                        if (df0[column_selected][i]){
                                                            if (key == "y"){
                                                                y_val = y_val - 1
                                                                df[key].push(y_val);
                                                            } else {
                                                                df[key].push(df0[key][i]); 
                                                            }
                                                        }
                                                    } else if (df0[column_selected][i] & (color_selected>0.5) & (df0[column_selected.replace("_check", "_suggestion")][i]=="")) {
                                                        if (key == "y"){
                                                            y_val = y_val - 1
                                                            df[key].push(y_val);
                                                        } else {
                                                            df[key].push(df0[key][i]); 
                                                        }
                                                    } else if (df0[column_selected][i] & (color_selected<=0.5) & (df0[column_selected.replace("_check", "_suggestion")][i]!="")) {
                                                        if (key == "y"){
                                                            y_val = y_val - 1
                                                            df[key].push(y_val);
                                                        } else {
                                                            df[key].push(df0[key][i]); 
                                                        }
                                                    }
                                                }
                                            }

                                        }
                                        source.change.emit();
                                        legend_source.change.emit();
                                    }
                                    console.log("Ended")
                                    cb_obj.indices = [];
                                    """))
        perc_plot.circle(x="x", y="y", size=9, line_width=0.5,
                         fill_color={"field": "color",
                                     "transform": CategoricalColorMapper(factors=[warning_tag, solved_tag],
                                                                         palette=["red", "orange"])},
                         fill_alpha="alpha",
                         source=legend_data)

        perc_data = np.array(perc_data)
        perc_data = ColumnDataSource(dict(
            x=perc_data[:, 0].astype(float),
            y=perc_data[:, 1].astype(float),
            perc=perc_data[:, 2]))

        image_perc = LabelSet(x="x", y="y", text="perc", source=perc_data, text_align="right", y_offset=0,
                              text_font_size="12px", text_baseline="middle")
        perc_plot.add_layout(image_perc)

        plot = column(perc_plot, row(data_table, check_plot))
        if self.latitude_tag:
            map_plot = self._create_map_plot(width + width_check, source)
            tabs = [TabPanel(child=plot, title="Details"), TabPanel(child=map_plot, title="Map")]
            plot = Tabs(tabs=tabs, tabs_location='left')

        plot = column(header, text_input, plot)

        if save_in_path:
            output_file(save_in_path, mode='inline')
            save(plot)
            os.startfile(save_in_path)
        else:
            show(plot)
