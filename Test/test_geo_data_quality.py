import logging
import unittest
from itertools import combinations

import geopandas as gpd
import pandas as pd
from shapely import Point

from geo_ita.src._data import get_df_comuni
from geo_ita.src._geo_data_quality import GeoDataQuality
import geo_ita.src.config as cfg


def sample_unique_by_column(df, column_name, sample_size=None):
    """
    Sample rows from a DataFrame, ensuring no duplicate values in a specific column,
    with rows randomly shuffled before selection.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        column_name (str): The column to ensure unique values.
        sample_size (int): The number of rows to sample.

    Returns:
        pd.DataFrame: A sampled DataFrame with unique values in the specified column.
    """
    shuffled_df = df.sample(frac=1, random_state=None).reset_index(drop=True)

    unique_rows = shuffled_df.drop_duplicates(subset=[column_name])

    if sample_size is None:
        sample_size = len(unique_rows)

    if len(unique_rows) < sample_size:
        raise ValueError("Sample size exceeds the number of unique values in the column.")

    sampled_df = unique_rows.sample(n=sample_size, random_state=None).reset_index(drop=True)
    return sampled_df


class TestGeoDataQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting logging
        logger = logging.getLogger("_data_enrichment")
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.DEBUG)

        # Load dataset just 1 time
        cls.df_comuni = get_df_comuni()

    def test_simple_run(self):
        df = self.df_comuni.sample(40)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=10)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"

        tags = [
            ("set_country_tag", cfg.TAG_COUNTRY),
            ("set_regioni_tag", cfg.TAG_REGIONE),
            ("set_province_tag", cfg.TAG_PROVINCIA),
            ("set_comuni_tag", cfg.TAG_COMUNE),
            ("set_latitude_longitude_tag", {"geometry_column": "geometry"})
        ]

        for k in range(1, len(tags) + 1):
            for perm in combinations(tags, k):
                # Check if "set_comuni_tag" is present then at least one between
                # "set_province_tag" and "set_regioni_tag" must be present.
                if ("set_comuni_tag", cfg.TAG_COMUNE) in perm:
                    if ("set_province_tag", cfg.TAG_PROVINCIA) not in perm and \
                            ("set_regioni_tag", cfg.TAG_REGIONE) not in perm:
                        continue
                dq = GeoDataQuality(df)
                for method, arg in perm:
                    if isinstance(arg, dict):
                        getattr(dq, method)(**arg)
                    else:
                        getattr(dq, method)(arg)
                dq.start_check()
                result = dq.get_results()
                self.assertEqual(0, len(result))

    def test_empty_run(self):
        df = sample_unique_by_column(self.df_comuni, cfg.TAG_REGIONE)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=1)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"

        tags = [
            ("set_country_tag", cfg.TAG_COUNTRY),
            ("set_regioni_tag", cfg.TAG_REGIONE),
            ("set_province_tag", cfg.TAG_PROVINCIA),
            ("set_comuni_tag", cfg.TAG_COMUNE),
            ("set_latitude_longitude_tag", {"geometry_column": "geometry"})
        ]

        for k in range(1, len(tags) + 1):
            for perm in combinations(tags, k):
                _df = df.copy()
                for j in range(0, k-1):
                    if perm[j][0] == "set_country_tag":
                        _df[perm[j][1]] = None
                    else:
                        _df[perm[j][1]] = ["wrong" if i % 2 == 0 else None for i in range(len(_df))]
                dq = GeoDataQuality(_df)
                for method, arg in perm:
                    if isinstance(arg, dict):
                        getattr(dq, method)(**arg)
                    else:
                        getattr(dq, method)(arg)
                dq.start_check()
                result = dq.get_results()
                if k == 1:
                    self.assertEqual(0, len(result))
                else:
                    self.assertEqual(len(df), len(result))
                    if (perm[-1] == tags[-2]) & ((tags[1] in perm) or (tags[2] in perm)):
                        # You cannot allways find the right provincia or regione in case of homonym comune
                        self.assertGreater(result["solved"].sum(), len(df) - 8)
                    else:
                        self.assertEqual(len(df), result["solved"].sum())
                        for j in range(0, k-1):
                            column = perm[j][1]
                            pd.testing.assert_series_equal(
                                dq._clean_denomination(df[column]),
                                result[column + dq.CORRECTION_SUFFIX],
                                check_names=False,
                            )

    def test_empty_run_distinct_correction(self):
        df = sample_unique_by_column(self.df_comuni, cfg.TAG_REGIONE, 16)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df = df.reset_index(drop=True)
        df.index = df.index % 8

        df_check = df.iloc[0:8][[cfg.TAG_PROVINCIA]]
        df_check[cfg.TAG_COMUNE] = df.iloc[8:16][cfg.TAG_COMUNE]
        df_check[cfg.TAG_REGIONE] = None

        dq = GeoDataQuality(df_check)
        dq.set_comuni_tag(cfg.TAG_COMUNE)
        dq.set_province_tag(cfg.TAG_PROVINCIA)
        dq.set_regioni_tag(cfg.TAG_REGIONE)
        dq.start_check()
        result = dq.get_results()
        self.assertEqual(len(df_check), len(result))
        self.assertEqual(0, result["solved"].sum())

    def test_plot_result_combination(self):
        df = self.df_comuni.sample(50)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=10)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"

        tags = [
            ("set_country_tag", cfg.TAG_COUNTRY),
            ("set_regioni_tag", cfg.TAG_REGIONE),
            ("set_province_tag", cfg.TAG_PROVINCIA),
            ("set_comuni_tag", cfg.TAG_COMUNE),
            ("set_latitude_longitude_tag", {"geometry_column": "geometry"})
        ]

        for k in range(2, len(tags) + 1):
            for perm in combinations(tags, k):
                _df = df.copy()
                for j in range(0, k - 1):
                    if perm[j][0] == "set_country_tag":
                        _df[perm[j][1]] = None
                    else:
                        _df[perm[j][1]] = ["wrong" if i % 2 == 0 else None for i in range(len(_df))]
                dq = GeoDataQuality(_df)
                for method, arg in perm:
                    if isinstance(arg, dict):
                        getattr(dq, method)(**arg)
                    else:
                        getattr(dq, method)(arg)
                dq.start_check()
                dq.plot_result(save_in_path="test_plot_result_combination.html")

    def test_plot_result(self):
        # Create a test DataFrame with different errors
        all_df = []
        # 1. Correct
        df = self.df_comuni.sample(5)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=1)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"
        df = df[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_COUNTRY, "geometry"]]
        all_df.append(df)

        # 2. Other Country
        df = pd.DataFrame(data=[["Madrid", "Madrid", "", "Spain", Point(40.41588398208069, -3.7015212072478687)]], columns=df.columns)

        # 3. Missing Country
        df = self.df_comuni.sample(2)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=1)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = None
        df = df[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_COUNTRY, "geometry"]]
        all_df.append(df)

        # 4. Missing Comune
        df = self.df_comuni.sample(2)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=1)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"
        df[cfg.TAG_COMUNE] = None
        df = df[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_COUNTRY, "geometry"]]
        all_df.append(df)

        # 5. Missing Coordinates
        df = self.df_comuni.sample(2)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=1)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"
        df["geometry"] = None
        df = df[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_COUNTRY, "geometry"]]
        all_df.append(df)

        # 5. Wrong Coordinates
        df = self.df_comuni.sample(2)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=1)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"
        df["geometry"] = df["geometry"].apply(lambda x: Point(x.y, x.x))
        df.loc[0, "geometry"] = Point(0, 0)
        df = df[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_COUNTRY, "geometry"]]
        all_df.append(df)


        # 6. Conflict
        df = sample_unique_by_column(self.df_comuni, cfg.TAG_REGIONE, 16)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df = df.reset_index(drop=True)
        df.index = df.index % 8
        df_check = df.iloc[0:8]
        df_check[cfg.TAG_PROVINCIA] = df.iloc[8:16][cfg.TAG_PROVINCIA]

        df = gpd.GeoDataFrame(df_check, geometry="geometry")
        df["points"] = df.sample_points(size=1)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        df[cfg.TAG_COUNTRY] = "Italy"
        df = df[[cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_COUNTRY, "geometry"]]
        all_df.append(df)

        all_df = pd.concat(all_df, ignore_index=True)

        dq = GeoDataQuality(all_df)
        dq.set_country_tag(cfg.TAG_COUNTRY)
        dq.set_regioni_tag(cfg.TAG_REGIONE)
        dq.set_province_tag(cfg.TAG_PROVINCIA)
        dq.set_comuni_tag(cfg.TAG_COMUNE)
        dq.set_latitude_longitude_tag(geometry_column="geometry")
        dq.start_check()
        dq.plot_result()
