import logging
import os
from enum import Enum
from pathlib import Path
from typing import Union, Optional

from bokeh.models import (
    ColumnDataSource, DataTable, TableColumn, HTMLTemplateFormatter, CategoricalColorMapper,
    LabelSet, Label, WheelZoomTool, CustomJS, TabPanel, Tabs, TextInput, HoverTool, Legend, LegendItem
)
from bokeh.plotting import save, figure
from bokeh.io import output_file, show
from bokeh.layouts import column, row
from valdec.decorators import validate
import pandas as pd
import numpy as np
from shapely import Point
import xyzservices.providers as xyz

from geo_ita.src._data_enrichment import AddGeographicalInfo, get_city_from_coordinates, _create_geo_dataframe, _get_margins
from geo_ita.src.utils import GeoLevel, CodeLevel, infer_geographical_category, get_tag_registry, \
    test_column_in_dataframe, Check
import geo_ita.src.config as cfg

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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
            self.original_df[self.POINT_COLUMN + "_x"] = pd.to_numeric(self.original_df[longitude_column],
                                                                       errors="coerce")
            self.original_df[self.POINT_COLUMN + "_y"] = pd.to_numeric(self.original_df[latitude_column],
                                                                       errors="coerce")
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
        check_df = addinfo.get_result(suffix=self.DATA_QUALITY_CHECK_TAG)
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
        check_df = addinfo.get_result(suffix=self.DATA_QUALITY_CHECK_TAG)
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
        check_df = addinfo.get_result(suffix=self.DATA_QUALITY_CHECK_TAG)
        self.original_df[cfg.TAG_REGIONE + "_" + GeoLevel.COMUNE] = self._clean_denomination(
            check_df[cfg.TAG_REGIONE + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_PROVINCIA + "_" + GeoLevel.COMUNE] = self._clean_denomination(
            check_df[cfg.TAG_PROVINCIA + self.DATA_QUALITY_CHECK_TAG])
        self.original_df[cfg.TAG_COMUNE + "_" + GeoLevel.COMUNE] = self._clean_denomination(
            check_df[cfg.TAG_COMUNE + self.DATA_QUALITY_CHECK_TAG])

        self._second_check(GeoLevel.COMUNE)

    def _check_coordinates(self):
        self._first_check(GeoLevel.COORDINATES)

        check_df = get_city_from_coordinates(self.original_df, suffix=self.DATA_QUALITY_CHECK_TAG)
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

        # TODO try invert the coordinates

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

    @validate
    def plot_result(
            self,
            background_color: str = "white",
            text_color: str = "black",
            title: str = "Geographical DataQuality",
            subtitle: str = None,
            save_in_path: Union[str, Path] = None
    ):
        self._prepare_plot_data()
        header = self._create_header(background_color, text_color, title, subtitle)
        text_input = self._create_text_key_copy()
        data_table, check_plot = self._create_data_table_and_check_plot(text_input)
        perc_plot = self._create_perc_plot()
        plot = column(perc_plot, row(data_table, check_plot))

        if GeoLevel.COORDINATES in self.detail_level:
            map_plot = self._create_map_plot()
            tabs = [TabPanel(child=plot, title="Details"), TabPanel(child=map_plot, title="Map")]
            plot = Tabs(tabs=tabs, tabs_location='left')

        plot = column(header, text_input, plot)
        self._save_plot(plot, save_in_path)

    def _prepare_plot_data(self):
        plot_data = self.original_df[self.original_df["check"]].copy()

        # Filter only column needed
        check_col = [a for a in plot_data.columns if self.CHECK_SUFFIX in a]
        propose_col = [a for a in plot_data.columns if self.CORRECTION_SUFFIX in a]
        original_columns = [v[0] for v in self.detail_level.values()]
        cross_columns = [
            f"{get_tag_registry(CodeLevel.DENOMINATION, list(self.detail_level.keys())[i])}_{list(self.detail_level.keys())[j]}"
            for i in range(len(self.detail_level))
            for j in range(i, len(self.detail_level))
        ]
        plot_data = plot_data[original_columns + check_col + propose_col + cross_columns + ["check", "solved"]]

        plot_data["x"] = 0.5
        plot_data["y"] = range(plot_data.shape[0])[::-1]
        plot_data["y"] += 0.5
        plot_data["selected_color"] = "transparent"
        plot_data["selected_alpha"] = 1
        plot_data["check_color"] = np.select(
            [~plot_data["check"], plot_data["check"] & ~plot_data["solved"], plot_data["solved"]],
            [Check.OK.value, Check.WARNING.value, Check.SOLVED.value],
        )
        if self.unique_key_column is None:
            self.unique_key_column = "index"
            plot_data = plot_data.reset_index().rename(columns={plot_data.index.name: self.unique_key_column})
        plot_data[self.unique_key_column] = plot_data[self.unique_key_column].astype(str)

        plot_data[propose_col] = plot_data[propose_col].fillna("")
        if GeoLevel.COORDINATES in self.detail_level:
            coord_column = self.detail_level[GeoLevel.COORDINATES][0]
            # Create GeoDataframe
            plot_data = _create_geo_dataframe(plot_data, geo_tag=coord_column)
            plot_data["coordinates"] = plot_data[coord_column].apply(
                lambda p: f"{p.x:.6f}-{p.y:.6f}" if p else None)
            plot_data = plot_data.to_crs({'init': "epsg:3857"})
            plot_data["longitudine_marcator"], plot_data[
                "latitudine_marcator"] = plot_data.geometry.x, plot_data.geometry.y
            plot_data = plot_data.to_crs({'init': "epsg:3857"})
            plot_data.drop(columns=["geometry"], inplace=True)
            plot_data = pd.DataFrame(plot_data)
            plot_data.rename(columns={"coordinates": coord_column}, inplace=True)
        self.plot_data = plot_data

    def _create_header(self, background_color, text_color, title, subtitle):
        height = 100 if subtitle else 50
        header = figure(x_range=(0, 1), y_range=(0, 1), width=1000, height=height, tools="")
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
            Label(x=0.005, y=.8, text=title, text_font_style="bold", text_font_size="20pt", text_baseline="top",
                  text_color=text_color))
        if subtitle:
            header.add_layout(
                Label(x=0.005, y=.4, text=subtitle, text_font_size="12pt", text_baseline="top", text_color=text_color))
        return header

    def _create_text_key_copy(self):
        return TextInput(value="", title=self.unique_key_column + ": ", width=200)

    def _create_data_table_and_check_plot(self, text_input):
        columns = self._create_table_columns()
        self.height = (self.plot_data.shape[0] + 1) * 30
        self.n_rows = self.plot_data.shape[0]
        self.original_source = ColumnDataSource(self.plot_data)
        self.source = ColumnDataSource(self.plot_data)
        data_table = DataTable(source=self.source, columns=columns, fit_columns=True, selectable=True,
                               sortable=False, editable=True, index_position=None, row_height=30,
                               height=self.height, width=1000)
        self.source.selected.js_on_change(
            'indices',
            CustomJS(args=dict(source=self.source,
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
            """.format(key=self.unique_key_column))
        )
        check_plot = self._create_check_plot()
        return data_table, check_plot

    def _create_table_columns(self):
        columns = [TableColumn(field=self.unique_key_column, title=self.unique_key_column,
                               formatter=HTMLTemplateFormatter(template=self._get_template()))]
        for level, v in self.detail_level.items():
            tag = get_tag_registry(CodeLevel.DENOMINATION, level)
            # if level != GeoLevel.COORDINATES:
            self._create_html_column(tag, v[0])
            columns.append(TableColumn(field=v[0] + "_html", title=v[0],
                                       formatter=HTMLTemplateFormatter(template=self._get_template(level, tag))))
            # else:
            #    columns.append(TableColumn(field=v[0], title=v[0],
            #                               formatter=HTMLTemplateFormatter(template=self._get_template(level, tag))))
        return columns

    def _create_html_column(self, column, original_column):
        pos = self.plot_data[column + self.CORRECTION_SUFFIX] != ""
        original = self.plot_data[original_column].fillna("NaN").copy()
        self.plot_data[original_column + "_html"] = np.where(pos,
                                                             original + "||" + self.plot_data[
                                                                 column + self.CORRECTION_SUFFIX],
                                                             " ||" + original)
        self.plot_data[original_column + "_html"] = self.plot_data[original_column + "_html"] + "||" + original + "||" + \
                                                    self.plot_data[
                                                        column + self.CORRECTION_SUFFIX]

        def add_html_detail(self, level):
            tag = f"{column}_{level}"
            if tag in self.plot_data.columns:
                self.plot_data[original_column + "_html"] = self.plot_data[original_column + "_html"] + "||" + \
                                                            self.plot_data[tag].fillna("-")

        for level in self.detail_level.keys():
            add_html_detail(self, level)
        return

    def _get_template(self, level=None, column=None):
        if column:
            return f"""
        <div style="background:<%=
            (function colorfromint(){{
                if({column + self.CHECK_SUFFIX}){{
                    if({column + self.CORRECTION_SUFFIX} != ""){{
                        return("orange")
                        }} else {{
                        return("red")
                    }}
                }}
            }}()) %>;
            color: black">
        <span href="#" data-toggle="tooltip" title="Original: <%= value.split('||')[2] %>\nSuggestion: <%= value.split('||')[3] %>{self._get_additional_tooltip(level)}">
            <strike><%=  value.split("||")[0] %></strike> <%= value.split("||")[1] %>
        </span>
        </div>
        """
        return """
        <div style="background:<%= selected_color %>; color: black">
        <%= value %>
        </div>
        """

    def _get_additional_tooltip(self, level):
        additional_tooltip = ""
        i = 4
        for level2 in [GeoLevel.COUNTRY, GeoLevel.REGIONE, GeoLevel.PROVINCIA, GeoLevel.COMUNE]:
            if (level2 in self.detail_level) & (level2 <= level):
                additional_tooltip += f"\n{str(level).capitalize()} found from {self.detail_level[level2][0]}: <%= value.split('||')[{i}] %>"
                i += 1
        return additional_tooltip

    def _create_check_plot(self):
        check_plot = figure(height=self.height, width=250, x_range=(0, 1), y_range=(0, self.n_rows),
                            x_axis_location="above", tools='')
        check_plot.xgrid.grid_line_color = None
        check_plot.ygrid.grid_line_color = None
        check_plot.yaxis.visible = False
        check_plot.grid.visible = False
        check_plot.toolbar.logo = None
        check_plot.outline_line_color = None
        check_plot.xaxis.major_label_text_font_size = '10pt'
        check_plot.xaxis.ticker = [0.5]
        check_plot.xaxis.major_label_overrides = {0.5: "Check"}
        color_mapper = CategoricalColorMapper(
            factors=[Check.OK.value, Check.WARNING.value, Check.SOLVED.value],
            palette=["green", "red", "orange"])
        check_plot.circle(x="x", y="y", size=9, line_width=0.5,
                          color={'field': 'check_color', 'transform': color_mapper}, source=self.source,
                          )
        legend_data = {
            'category': [Check.OK.value, Check.WARNING.value, Check.SOLVED.value],
            'color': ["green", "red", "orange"],
            'alpha': [1, 1, 1],
            'x': [1, 2, 3],
            'y': [-1, -1, -1],
        }
        legend_source = ColumnDataSource(legend_data)

        legend_renderers = check_plot.circle(
            x='x', y='y', source=legend_source,
            size=10, color='color', alpha='alpha',
        )

        # Creazione degli elementi della legenda
        legend = Legend(items=[
            LegendItem(label=dict(field="category"), renderers=[legend_renderers])
        ])
        check_plot.add_layout(legend, 'right')
        return check_plot

    def _create_perc_plot(self, ):
        perc_data, legend_data = self._prepare_perc_and_legend_data()
        perc_plot = figure(height=50, width=1000, x_range=(0, len(self.detail_level)), y_range=(0, 1), tools='tap')
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
        legend_data = ColumnDataSource(
            dict(x=legend_data[:, 0].astype(float), y=legend_data[:, 1].astype(float), color=legend_data[:, 2],
                 column=legend_data[:, 3], alpha=np.ones(legend_data.shape[0]) * 0.5))
        legend_data.selected.js_on_change(
            'indices',
            CustomJS(args=dict(source=self.source,
                               original_source=self.original_source,
                               legend_source=legend_data), code=f"""
            var indices = cb_obj.indices;
            if (indices.length > 0){{
                var df_legend = legend_source.data;
                var pos = cb_obj.indices[0];
                console.log("Selected", pos) 
                var data = source.data;
                var column_selected = df_legend["column"][pos];
                var color_selected = df_legend["y"][pos];
                var previous_selected = (df_legend["alpha"][pos] == 1);
                console.log("Previous selected", previous_selected) 
                var df0 = original_source.data;
                var df = source.data;
                if (previous_selected){{
                    df_legend["alpha"][pos] = 0.5
                    for (var key in df0) {{
                        df[key] = [];
                        for (var i = 0; i < df0[key].length; ++i) {{
                            df[key].push(df0[key][i]);
                        }}
                    }}
                }} else {{
                    df_legend["alpha"][pos] = 1
                    for (var key in df0) {{
                        var y_val = df0[key].length + 0.5
                        df[key] = [];
                        for (i = 0; i < df0[key].length;i++){{
                            if (column_selected.includes("coordinates")) {{
                                if (df0[column_selected][i]){{
                                    if (key == "y"){{
                                        y_val = y_val - 1
                                        df[key].push(y_val);
                                    }} else {{
                                        df[key].push(df0[key][i]); 
                                    }}
                                }}
                            }} else if (df0[column_selected][i] & (color_selected>0.5) & 
                            (df0[column_selected.replace("{self.CHECK_SUFFIX}", "{self.CORRECTION_SUFFIX}")][i]=="")) {{
                                if (key == "y"){{
                                    y_val = y_val - 1
                                    df[key].push(y_val);
                                }} else {{
                                    df[key].push(df0[key][i]); 
                                }}
                            }} else if (df0[column_selected][i] & (color_selected<=0.5) & 
                            (df0[column_selected.replace("{self.CHECK_SUFFIX}", "{self.CORRECTION_SUFFIX}")][i]!="")) {{
                                if (key == "y"){{
                                    y_val = y_val - 1
                                    df[key].push(y_val);
                                }} else {{
                                    df[key].push(df0[key][i]); 
                                }}
                            }}
                        }}
                    }}
                }}
                source.change.emit();
                legend_source.change.emit();
            }}
            console.log("Ended")
            cb_obj.indices = [];
            """))
        perc_plot.circle(x="x", y="y", size=9, line_width=0.5, fill_color={"field": "color",
                                                                           "transform": CategoricalColorMapper(
                                                                               factors=[Check.WARNING.value,
                                                                                        Check.SOLVED.value],
                                                                               palette=["red", "orange"])},
                         fill_alpha="alpha", source=legend_data)
        perc_data = ColumnDataSource(
            dict(x=perc_data[:, 0].astype(float), y=perc_data[:, 1].astype(float), perc=perc_data[:, 2]))
        perc_plot.add_layout(
            LabelSet(x="x", y="y", text="perc", source=perc_data, text_align="right", y_offset=0, text_font_size="12px",
                     text_baseline="middle"))
        return perc_plot

    def _prepare_perc_and_legend_data(self):
        perc_data = []
        legend_data = []
        i = 0
        for level, v in self.detail_level.items():
            tag = get_tag_registry(CodeLevel.DENOMINATION, level)
            check_pos = self.plot_data[tag + self.CHECK_SUFFIX]
            if level != GeoLevel.COORDINATES:
                check_pos = check_pos & (self.plot_data[tag + self.CORRECTION_SUFFIX] == "")
            n_check = check_pos.sum()
            perc_data.append(
                [i + 0.8, 0.75, "{} ({}%)".format(n_check, int(round(n_check / self.plot_data.shape[0] * 100, 0)))])
            legend_data.append([i + 0.9, 0.75, Check.WARNING.value, tag + self.CHECK_SUFFIX])
            if level != GeoLevel.COORDINATES:
                n_propose = (self.plot_data[tag + self.CORRECTION_SUFFIX] != "").sum()
                perc_data.append(
                    [i + 0.8, 0.25,
                     "{} ({}%)".format(n_propose, int(round(n_propose / self.plot_data.shape[0] * 100, 0)))])
                legend_data.append([i + 0.9, 0.25, Check.SOLVED.value, tag + self.CHECK_SUFFIX])
            i += 1
        return np.array(perc_data), np.array(legend_data)

    def _create_map_plot(self):
        margins, shape = _get_margins()

        map_plot = figure(x_range=(margins[0][0], margins[0][1]),
                          y_range=(margins[1][0], margins[1][1]),
                          x_axis_type="mercator", y_axis_type="mercator", width=1250, tools='pan,tap,wheel_zoom')
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
        map_plot.toolbar_location = None
        color_mapper = CategoricalColorMapper(
            factors=[Check.OK.value, Check.WARNING.value, Check.SOLVED.value],
            palette=["green", "red", "orange"])
        plot1 = map_plot.circle(x="longitudine_marcator", y="latitudine_marcator", size=7, fill_alpha="selected_alpha",
                                line_color="gray", line_width=0.5, source=self.source, legend_field="check_color",
                                color={'field': 'check_color', 'transform': color_mapper})
        tooltips = [(self.unique_key_column, "@" + self.unique_key_column)]
        if GeoLevel.COUNTRY in self.detail_level:
            tooltips.append(("Nazione", "@" + self.detail_level[GeoLevel.COUNTRY][0]))
        if GeoLevel.REGIONE in self.detail_level:
            tooltips.append(("Regione", "@" + self.detail_level[GeoLevel.REGIONE][0]))
        if GeoLevel.PROVINCIA in self.detail_level:
            tooltips.append(("Provincia", "@" + self.detail_level[GeoLevel.PROVINCIA][0]))
        if GeoLevel.COMUNE in self.detail_level:
            tooltips.append(("Comune", "@" + self.detail_level[GeoLevel.COMUNE][0]))
        tooltips.append(("Coordinates", "@" + self.detail_level[GeoLevel.COORDINATES][0]))
        map_plot.add_tools(HoverTool(renderers=[plot1], tooltips=tooltips))
        map_plot.toolbar.active_scroll = map_plot.select_one(WheelZoomTool)
        return map_plot

    def _save_plot(self, plot, save_in_path):
        if save_in_path:
            output_file(save_in_path, mode='inline')
            save(plot)
        else:
            temp_path = os.path.join(os.path.expanduser("~"), "temp_plot.html")
            output_file(temp_path, mode='inline')
            save(plot)
            os.startfile(temp_path)
