import pandas as pd

from geo_ita.src._data import *
from geo_ita.src._data import upload_data_istat_from_api
from geo_ita.src.definition import *

import unittest
import logging
from pathlib import PureWindowsPath


class TestEnrichDataframe(unittest.TestCase):

    def test_get_df_comuni(self):
        df = get_df_comuni()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)

    def test_get_df_province(self):
        df = get_df_province()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)

    def test_get_df_regioni(self):
        df = get_df_regioni()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)

    def test_get_list_comuni(self):
        result = get_list_comuni()
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

    def test_get_list_province(self):
        result = get_list_province()
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

    def test_get_list_regioni(self):
        result = get_list_regioni()
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

    def test_download_high_density_population_df(self):
        remove_high_resolution_population_density_file()
        df = get_high_resolution_population_density_df()
        del df
        df = get_high_resolution_population_density_df()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)

    def test_upload_data_istat_from_api(self):
        upload_data_istat_from_api()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
