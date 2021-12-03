from pandas._testing import assert_frame_equal

from geo_ita.src._enrich_dataframe import __find_coordinates_system

from geo_ita.src._enrich_dataframe import *
from geo_ita.src._data import *
from geo_ita.src.definition import *
from pathlib import PureWindowsPath
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import unittest


class TestEnrichDataframe(unittest.TestCase):

    # get_coordinates_from_address

    def xtest_get_coordinates_from_address_input(self):
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

    def xtest_get_coordinates_from_address_match(self):
        df, address = pd.DataFrame(data=[["Corso di Francia Roma"]], columns=["address"]), "address"
        result = get_coordinates_from_address(df, address)
        result = get_city_from_coordinates(result)
        self.assertEqual("Roma", result["denominazione_comune"].values[0])
        df, address = pd.DataFrame(data=[["Corso di Francia Roma", "Firenze"]], columns=["address", "city"]), "address"
        city = "city"
        result = get_coordinates_from_address(df, address, city)
        self.assertEqual(None, result["latitude"].values[0])
        df, address = pd.DataFrame(data=[["Corso di Francia Roma", "Firenze"],
                                         ["Corso di Francia", "Roma"],
                                         ["Viale G. P. da Palestrina", "Latina"],
                                         ["Via S. Barbara", "Paola"],
                                         ["Via dellAquila Reale", "Roma"],
                                         ["xxxx", None]], columns=["address", "city"]), "address"
        city = "city"
        result = get_coordinates_from_address(df, address, city)
        result = get_city_from_coordinates(result)
        self.assertTrue(pd.isnull(result["latitude"].values[0]))
        self.assertTrue(pd.isnull(result["latitude"].values[-1]))
        self.assertEqual("Roma", result["denominazione_comune"].values[1])
        self.assertEqual("Roma", result["denominazione_comune"].values[4])

    # get_address_from_coordinates

    def xtest_get_address_from_coordinates_input(self):
        df = ["via corso di Francia"]
        with self.assertRaises(Exception):
            get_address_from_coordinates(df)
        df = pd.DataFrame(data=[["via corso di Francia"]], columns=["address"])
        with self.assertRaises(Exception):
            get_address_from_coordinates(df)
        df, latitude_columns, longitude_columns = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]], columns=["latitude", "longitude"]), \
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

    def xtest_get_address_from_coordinates_match(self):
        df = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]], columns=["lat", "lon"])
        result = get_address_from_coordinates(df)
        self.assertEqual("Roma", result["city"].values[0])
        df = pd.DataFrame(data=[[41.93683317516326, 12.471707219950744]], columns=["lat", "lon"])
        result = get_address_from_coordinates(df, latitude_columns="lat", longitude_columns="lon")
        self.assertEqual("Roma", result["city"].values[0])
        df = pd.DataFrame(data=[[43.884609765796114, 8.8971202373737]], columns=["lat", "lon"])
        result = get_address_from_coordinates(df)
        self.assertEqual(None, result["city"].values[0])

    # AddGeographicalInfo

    def xtest_add_geographical_info_input(self):
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
            addinfo.get_not_matched()

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
        addinfo.run_find_frazioni()
        addinfo.run_find_frazioni_from_google()
        addinfo.run_similarity_match()
        addinfo.use_manual_match({"rome": "roma"})
        result = addinfo.get_result()
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(0, result.shape[0])
        self.assertCountEqual(['city', cfg.TAG_COMUNE, cfg.TAG_CODICE_COMUNE,
                                       cfg.TAG_PROVINCIA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
                                       cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE,
                                       cfg.TAG_AREA_GEOGRAFICA,
                                       cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE], list(result.columns))

    def xtest_add_geographical_info_match(self):
        df = pd.DataFrame(data=[["Milano", "Milano", "Milano", "MI", "Lombardia"],
                                ["florence", "Firenze", "Firenze", "FI", "Toscana"],
                                ["porretta terme", "Alto Reno Terme", "Bologna", "BO", "Emilia-Romagna"],
                                ["marina di ardea", "Ardea", "Roma", "RM", "Lazio"],
                                ["Baranzate", "Baranzate", "Milano", "MI", "Lombardia"],
                                ["milano marittima", "Cervia", "Ravenna", "RA", "Emilia-Romagna"],
                                ["xxxx", None, None, None, None],
                                [None, None, None, None, None]],
                          columns=["Citta", "comune", "provincia", "sl", "regione"])

        addinfo = AddGeographicalInfo(df)
        addinfo.set_comuni_tag("Citta")
        addinfo.run_simple_match()
        addinfo.run_find_frazioni()
        addinfo.run_find_frazioni_from_google()
        result = addinfo.get_result()
        result = result.where(pd.notnull(result), None)
        self.assertTrue(result["denominazione_comune"].equals(result["comune"]))
        self.assertTrue(result["sigla"].equals(result["sl"]))
        self.assertTrue(result["denominazione_regione"].equals(result["regione"]))

    def xtest_aggregate_point_by_distance(self):
        df = get_df_comuni()
        df = aggregate_point_by_distance(df, 5000, latitude_columns="center_y", longitude_columns="center_x")

    def xtest_get_population_nearby_input(self):
        pass

    def xtest_get_population_nearby_usage(self):
        test_df = pd.DataFrame([[41.65756068387786, 13.351242360288134]], columns=["center_y", "center_x"])
        #test_df = get_df_comuni()
        test_df = get_population_nearby(test_df, 300, latitude_columns="center_y", longitude_columns="center_x")
        prova = ""

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
        dq.plot_result()
        col_test = ["nazione", "regione", "provincia", "comune",
                    "nazione_check", "nazione_suggestion", "regione_check", "regione_suggestion",
                    "provincia_check", "provincia_suggestion", "comune_check", "comune_suggestion",
                    "coordinates_check", "check", "solved"]
        assert_frame_equal(result[col_test],
                           df[col_test],
                           check_names=False, check_dtype=False
                           )
        prova = ""


def test_KDEDensity():
    test_df = pd.read_pickle(
        r"C:\Users\A470222\Documents\Python Scripts\ex_mobility\data\Geo/Densita\population_ita_2019-07-01.pkl")
    test_df = test_df[test_df["denominazione_comune"] == "Prato"]

    kde = KDEDensity(test_df, "Lat", "Lon", value_tag="Population")
    prova1 = kde.evaluate_in_point(43.89243338039644, 11.07762361613304)
    prova2 = kde.evaluate_in_point(43.874280169137386, 11.065771494973662)
    prova = "Ciao"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

    #test_KDEDensity()

