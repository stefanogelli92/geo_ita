import pandas as pd

from geo_ita.src._data import *
from geo_ita.src._data import update_data_istat
from geo_ita.src.definition import *

import unittest
import logging
from pathlib import PureWindowsPath


class TestData(unittest.TestCase):

    def test_get_df_comuni(self):
        df = get_df_comuni()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)
        non_emphty_columns = [cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]
        self.assertTrue(set(df.columns).issuperset(set(non_emphty_columns)))
        for col in non_emphty_columns:
            pos = df[col].notnull()
            if pos.sum() > 0:
                log.warning(f"{col} not found for comuni: {','.join(df[pos][cfg.TAG_COMUNE].values)}")
            self.assertGreater(pos.mean(), 0.1)

    def test_get_df_province(self):
        df = get_df_province()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)
        non_emphty_columns = [cfg.TAG_PROVINCIA, cfg.TAG_REGIONE]
        numeric_columns = [cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]
        non_emphty_columns = non_emphty_columns + numeric_columns
        self.assertTrue(set(df.columns).issuperset(set(non_emphty_columns)))
        for col in non_emphty_columns:
            self.assertEqual(df[col].isna().sum(), 0)
        for col in numeric_columns:
            self.assertEqual((df[col] <= 0).sum(), 0)

    def test_get_df_regioni(self):
        df = get_df_regioni()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)
        non_emphty_columns = [cfg.TAG_REGIONE]
        numeric_columns = [cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]
        non_emphty_columns = non_emphty_columns + numeric_columns
        self.assertTrue(set(df.columns).issuperset(set(non_emphty_columns)))
        for col in non_emphty_columns:
            self.assertEqual(df[col].isna().sum(), 0)
        for col in numeric_columns:
            self.assertEqual((df[col] <= 0).sum(), 0)

    def test_get_comuni_list(self):
        result = get_comuni_list()
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

    def test_get_province_list(self):
        result = get_province_list()
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

    def test_get_regioni_list(self):
        result = get_regioni_list()
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

    def test_download_high_density_population_df(self):
        remove_high_resolution_population_density_file()
        df = get_high_resolution_population_density_df()
        del df
        df = get_high_resolution_population_density_df()
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)

    def test_update_data_istat(self):
        logging.basicConfig(level=logging.INFO)
        update_data_istat(year=2022)
        df = get_df_comuni()
        n_population_2022 = 59019317.0  # 58991941.0
        self.assertEqual(df[cfg.TAG_POPOLAZIONE].sum(), n_population_2022)
        update_data_istat()
        df = get_df_comuni()
        self.assertNotEqual(df[cfg.TAG_POPOLAZIONE].sum(), n_population_2022)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
