from pandas.testing import assert_frame_equal, assert_series_equal

from geo_ita.src._data_enrichment import *
from geo_ita.src._data import *
from geo_ita.src.definition import *
from pathlib import PureWindowsPath
from geo_ita.src.config import *
import logging

log = logging.getLogger("_data_enrichment")
log.addHandler(logging.NullHandler())
log.setLevel(logging.DEBUG)

import unittest


class TestDataEnrichment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load dataset just 1 time
        cls.df_comuni = get_df_comuni()

    @staticmethod
    def check_result_on_same_dataframe(df, test_columns, result_columns=None, suffix: str = None):
        df1 = df[test_columns]
        if result_columns is not None:
            df2 = df[result_columns]
            df2.columns = test_columns
        elif suffix is not None:
            df2 = df[[col + suffix for col in test_columns]]
            df2.columns = test_columns

        assert_frame_equal(df1, df2)


class TestGetCoordinatesFromAddress(unittest.TestCase):

    def test_get_coordinates_from_address_input(self):
        # Use wrong input type
        df, address = ["via corso di Francia"], "address"
        with self.assertRaises(Exception):
            get_coordinates_from_address(df, address)
        df, address = pd.DataFrame(data=[["via corso di Francia"]], columns=["address"]), ["address"]
        with self.assertRaises(Exception):
            get_coordinates_from_address(df, address)
        df, address = pd.DataFrame(data=[["via corso di Francia"]], columns=["address"]), "addres"
        with self.assertRaises(Exception):
            get_coordinates_from_address(df, address)
        # Empthy Dataframe
        df, address = pd.DataFrame(columns=["address"]), "address"
        result = get_coordinates_from_address(df, address)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(0, result.shape[0])
        self.assertListEqual(["address", "latitude", "longitude"], list(result.columns))

    def test_get_coordinates_from_address_match(self):
        df, address = pd.DataFrame(data=[["Corso di Francia Roma"]], columns=["address"]), "address"
        result = get_coordinates_from_address(df, address)
        result = get_city_from_coordinates(result)
        self.assertEqual("Roma", result[TAG_COMUNE].values[0])
        df, address = pd.DataFrame(data=[["Corso di Francia Roma", "Firenze"]], columns=["address", "city"]), "address"
        city = "city"
        result = get_coordinates_from_address(df, address, city)
        self.assertEqual(None, result["latitude"].values[0])
        df, address = pd.DataFrame(data=[["Corso di Francia Roma", "Firenze", None],
                                         ["Corso di Francia", "Roma", "Roma"],
                                         ["Viale G. P. da Palestrina", "Latina", "Latina"],
                                         ["Via dellAquila Reale", "Roma", "Roma"],
                                         ["xxxx", None, None]], columns=["address", "city", "comune_check"]), "address"
        city = "city"
        result = get_coordinates_from_address(df, address, city)
        result = get_city_from_coordinates(result)
        result.loc[result[TAG_PROVINCIA].isnull(), TAG_PROVINCIA] = None
        assert_series_equal(result["comune_check"],
                            result[TAG_PROVINCIA],
                            check_names=False, check_dtype=False
                            )


class TestGetAddressFromCoordinates(unittest.TestCase):

    def test_get_address_from_coordinates_input(self):
        df = ["via corso di Francia"]
        with self.assertRaises(Exception):
            get_address_from_coordinates(df)
        df = pd.DataFrame(data=[["via corso di Francia"]], columns=["address"])
        with self.assertRaises(Exception):
            get_address_from_coordinates(df)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]],
                                                               columns=["latitude", "longitude"]), \
            "lat", "lon"
        with self.assertRaises(Exception):
            get_address_from_coordinates(df, latitude_columns=latitude_columns, longitude_columns=longitude_columns)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[["A", "B"]],
                                                               columns=["latitude", "longitude"]), \
            "latitude", "longitude"
        with self.assertRaises(Exception):
            get_address_from_coordinates(df, latitude_columns=latitude_columns, longitude_columns=longitude_columns)
        # Empthy Dataframe
        df = pd.DataFrame(columns=["lat", "lon"])
        result = get_address_from_coordinates(df)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(0, result.shape[0])
        self.assertListEqual(['lat', 'lon', 'address', 'city'], list(result.columns))

    def test_get_address_from_coordinates_results(self):
        df = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]], columns=["lat", "lon"])
        result = get_address_from_coordinates(df)
        self.assertEqual("Roma", result["city"].values[0])
        df = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]], columns=["lat", "lon"])
        result = get_address_from_coordinates(df, latitude_columns="lat", longitude_columns="lon")
        self.assertEqual("Roma", result["city"].values[0])
        df = pd.DataFrame(data=[[43.884609765796114, 8.8971202373737]], columns=["lat", "lon"])
        result = get_address_from_coordinates(df)
        self.assertEqual(None, result["city"].values[0])


class TestGetCityFromCoordinates(unittest.TestCase):

    def test_get_city_from_coordinates_input(self):
        df = ["via corso di Francia"]
        with self.assertRaises(Exception):
            get_city_from_coordinates(df)
        df = pd.DataFrame(data=[["via corso di Francia"]], columns=["address"])
        with self.assertRaises(Exception):
            get_city_from_coordinates(df)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]],
                                                               columns=["latitude", "longitude"]), \
            "lat", "lon"
        with self.assertRaises(Exception):
            get_city_from_coordinates(df, latitude_columns=latitude_columns, longitude_columns=longitude_columns)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[["A", "B"]],
                                                               columns=["latitude", "longitude"]), \
            "latitude", "longitude"
        with self.assertRaises(Exception):
            get_city_from_coordinates(df, latitude_columns=latitude_columns, longitude_columns=longitude_columns)
        # Empthy Dataframe
        df = pd.DataFrame(columns=["lat", "lon"])
        result = get_city_from_coordinates(df)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(0, result.shape[0])
        self.assertListEqual(['lat', 'lon', TAG_COMUNE, TAG_PROVINCIA, TAG_SIGLA, TAG_REGIONE], list(result.columns))

    def test_get_city_from_coordinates_results(self):
        df = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]], columns=["lat", "lon"])
        result = get_city_from_coordinates(df)
        self.assertEqual("Roma", result[TAG_COMUNE].values[0])
        df = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]], columns=["lat", "lon"])
        result = get_city_from_coordinates(df, latitude_columns="lat", longitude_columns="lon")
        self.assertEqual("Roma", result[TAG_COMUNE].values[0])
        df = pd.DataFrame(data=[[43.884609765796114, 8.8971202373737]], columns=["lat", "lon"])
        result = get_city_from_coordinates(df)
        self.assertTrue(result[TAG_COMUNE].isna().all())


class TestAddGeographicalInfo(TestDataEnrichment):

    def test_add_geographical_info_input(self):
        df = ["via corso di Francia"]
        with self.assertRaises(Exception):
            AddGeographicalInfo(df)
        df = pd.DataFrame(data=[["roma"]], columns=["city"])
        comune_column = "comune"
        addinfo = AddGeographicalInfo(df)
        with self.assertRaises(Exception):
            addinfo.set_comuni_tag(comune_column)
        with self.assertRaises(Exception):
            addinfo.set_province_tag(comune_column)
        with self.assertRaises(Exception):
            addinfo.set_regioni_tag(comune_column)
        with self.assertRaises(Exception):
            addinfo.run_simple_match()
        with self.assertRaises(Exception):
            addinfo.get_not_matched_list()

        df = pd.DataFrame(data=[["Roma"]], columns=["city"])
        comune_column = "city"
        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag(comune_column)
        with self.assertRaises(Exception):
            addinfo.get_result()
        addinfo.run_simple_match()
        with self.assertRaises(Exception):
            addinfo.use_manual_match("roma")
        with self.assertRaises(Exception):
            addinfo.use_manual_match(["roma"])

        df = pd.DataFrame(columns=["city"])
        comune_column = "city"
        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag(comune_column)
        addinfo.run_simple_match()
        # addinfo.run_find_frazioni()
        # addinfo.run_find_frazioni_from_google()
        # addinfo.run_similarity_match()
        # addinfo.use_manual_match({"rome": "roma"})
        result = addinfo.get_result()
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(0, result.shape[0])
        self.assertCountEqual(['city'] + addinfo.OUTPUT_COLUMNS, list(result.columns))

    def test_simple_match(self):
        df = self.df_comuni.copy()

        addinfo = AddGeographicalInfo(df.sample(400))
        addinfo.set_comuni_tag(cfg.TAG_CODICE_COMUNE)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

        addinfo = AddGeographicalInfo(df.drop(columns=[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]).sample(400))
        addinfo.set_province_tag(cfg.TAG_CODICE_PROVINCIA)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

        addinfo = AddGeographicalInfo(df.drop(columns=[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]).sample(400))
        addinfo.set_regioni_tag(cfg.TAG_CODICE_REGIONE)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

        addinfo = AddGeographicalInfo(df.drop(columns=[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]).sample(400))
        addinfo.set_regioni_tag(cfg.TAG_REGIONE)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

        addinfo = AddGeographicalInfo(df.drop(columns=[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]).sample(400))
        addinfo.set_province_tag(cfg.TAG_PROVINCIA)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

        addinfo = AddGeographicalInfo(df.drop(columns=[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]).sample(400))
        addinfo.set_province_tag(cfg.TAG_SIGLA)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

        addinfo = AddGeographicalInfo(df.reset_index(drop=True))
        addinfo.original_df["key"] = addinfo.original_df[cfg.TAG_COMUNE]
        addinfo.original_df.loc[0, "key"] += r"/"
        addinfo.original_df.loc[1, "key"] += " "
        addinfo.original_df.loc[2, "key"] += "-"
        addinfo.original_df.loc[3, "key"] = "comune di " + addinfo.original_df.loc[3, "key"]
        addinfo.set_comuni_tag("key")
        addinfo.set_province_tag(cfg.TAG_SIGLA)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

    def test_bilingual_name(self):
        df = self.df_comuni.copy()
        df = df[df[cfg.TAG_COMUNE] != df[cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA]]
        df = df.sample(40)
        df[cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA].iloc[:35] = df[cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA].str.split(
            f"/").iloc[:35]
        df = df.explode(cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA)
        df = df[df[cfg.TAG_COMUNE] != df[cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA]]
        df = df[~df[cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA].isin(self.df_comuni[cfg.TAG_COMUNE])]

        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag(cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

    def test_find_any_variation_from_istat_history(self):
        df = get_administrative_changes_df()
        registry = self.df_comuni
        is_comune_now = df[cfg.TAG_CODICE_COMUNE].isin(registry[cfg.TAG_CODICE_COMUNE].values)
        df = pd.concat([df[is_comune_now].sample(40), df[~is_comune_now].sample(40)])
        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag(cfg.TAG_COMUNE)
        addinfo.run_simple_match()
        result = addinfo.get_result(handle_duplicate_column="_test")
        result["check"] = result[cfg.TAG_COMUNE].isna()
        self.assertEqual(0, result["check"].sum())

    def test_run_find_frazioni(self):
        df = pd.DataFrame(data=[
            ["Lido di Ostia", "Roma", "Roma", "RM", "Lazio"],
            ["Polesio", "Ascoli Piceno", "Ascoli Piceno", "AP", "Marche"],
            ["Carnaiola", "Fabro", "Terni", "TR", "Umbria"],
            ["xxx", None, None, None, None],
            ["Barcellona", None, None, None, None],
        ],
            columns=["Citta", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]
        )
        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag("Citta")
        addinfo.run_simple_match()
        addinfo.run_find_frazioni()
        result = addinfo.get_result(handle_duplicate_column="_test")
        result = result.where(pd.notnull(result), None)
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

    def test_run_find_frazioni_on_web(self):
        df = pd.DataFrame(data=[
            ["Lido di Ostia", "Roma", "Roma", "RM", "Lazio"],
            ["Polesio", "Ascoli Piceno", "Ascoli Piceno", "AP", "Marche"],
            ["Carnaiola", "Fabro", "Terni", "TR", "Umbria"],
            ["xxx", None, None, None, None],
            ["Barcellona", None, None, None, None],
        ],
            columns=["Citta", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]
        )
        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag("Citta")
        addinfo.run_simple_match()
        addinfo.run_find_frazioni_on_web()
        result = addinfo.get_result(handle_duplicate_column="_test")
        result = result.where(pd.notnull(result), None)
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")

    def test_run_similarity_match(self):
        df = pd.DataFrame(data=[
            ["Ascoli Picano", "Ascoli Piceno", "Ascoli Piceno", "AP", "Marche"],
            ["Reggio Calabbria", "Reggio di Calabria", "Reggio Calabria", "RC", "Calabria"],
            ["Trenii", None, None, None, None],
            ["Barcellona", None, None, None, None]
        ],
            columns=["Citta", cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_SIGLA, cfg.TAG_REGIONE]
        )
        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag("Citta")
        addinfo.run_simple_match()
        addinfo.run_similarity_match()
        addinfo.accept_similarity_result()
        result = addinfo.get_result(handle_duplicate_column="_test")
        result = result.where(pd.notnull(result), None)
        column_test = [column.replace("_test", "") for column in result.columns if "_test" in column]
        self.check_result_on_same_dataframe(result, column_test, suffix="_test")


class TestGetPopulationNearby(unittest.TestCase):

    def test_get_population_nearby_input(self):
        df = ["via corso di Francia"]
        with self.assertRaises(Exception):
            get_population_nearby(df, 500)
        df = pd.DataFrame(data=[["via corso di Francia"]], columns=["address"])
        with self.assertRaises(Exception):
            get_population_nearby(df, 500)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]],
                                                               columns=["latitude", "longitude"]), \
            "lat", "lon"
        with self.assertRaises(Exception):
            get_population_nearby(df, 100, latitude_columns=latitude_columns, longitude_columns=longitude_columns)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[["A", "B"]],
                                                               columns=["latitude", "longitude"]), \
            "latitude", "longitude"
        with self.assertRaises(Exception):
            get_population_nearby(df, 100, latitude_columns=latitude_columns, longitude_columns=longitude_columns)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]],
                                                               columns=["latitude", "longitude"]), \
            "latitude", "longitude"
        with self.assertRaises(Exception):
            get_population_nearby(df, "raggio", latitude_columns=latitude_columns, longitude_columns=longitude_columns)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]],
                                                               columns=["latitude", "longitude"]), \
            "latitude", "longitude"
        with self.assertRaises(Exception):
            get_population_nearby(df, 0, latitude_columns=latitude_columns, longitude_columns=longitude_columns)
            # Empthy Dataframe
        df = pd.DataFrame(columns=["lat", "lon"])
        result = get_population_nearby(df, 500)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(0, result.shape[0])
        self.assertListEqual(['lat', 'lon', 'n_residents'],
                             list(result.columns))

    def test_get_population_nearby_results(self):
        test_df = pd.DataFrame([[41.8343354636729, 12.4684276148718],
                                [42.23774542118423, 11.961695397335165]], columns=["center_y", "center_x"])
        test_df = get_population_nearby(test_df, 300, latitude_columns="center_y", longitude_columns="center_x")
        self.assertGreater(test_df["n_residents"].values[0], 100)
        self.assertEqual(test_df["n_residents"].values[1], 0)


class Prova(unittest.TestCase):

    def test_aggregate_point_by_distance(self):
        df = get_df_comuni()
        df = aggregate_point_by_distance(df, 5000, latitude_columns="center_y", longitude_columns="center_x")

    # GeoDataQuality

    def test_GeoDataQuality(self):
        df = pd.read_excel(root_path / PureWindowsPath(r"data_sources/Test/data_quality_samples.xlsx"))
        dq = GeoDataQuality(df)
        dq.set_nazione_tag("nazione")
        dq.set_regioni_tag("regione")
        dq.set_province_tag("provincia")
        dq.set_comuni_tag("comune", use_for_check_nation=True)
        dq.set_latitude_longitude_tag("latitudine", "longitudine")
        result = dq.start_check(show_only_warning=False, sensitive=True)
        # dq.plot_result()
        col_test = ["nazione", "regione", "provincia", "comune",
                    "nazione_check", "nazione_suggestion", "regione_check", "regione_suggestion",
                    "provincia_check", "provincia_suggestion", "comune_check", "comune_suggestion",
                    "coordinates_check", "check", "solved"]
        assert_frame_equal(result[col_test],
                           df[col_test],
                           check_names=False, check_dtype=False
                           )
