import unittest
import logging
import pandas as pd
from geo_ita.src._data import (
    get_df_comuni, get_df_province, get_df_regioni,
    get_comuni_list, get_province_list, get_regioni_list,
    get_high_resolution_population_density_df, remove_high_resolution_population_density_file,
    update_data_istat
)

import geo_ita.src.config as cfg


class TestData(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    def test_update_data_istat(self):
        logging.basicConfig(level=logging.INFO)
        update_data_istat(year=2022)
        df = get_df_comuni()
        n_population_2022 = 59019317.0
        self.assertEqual(n_population_2022, df[cfg.TAG_POPOLAZIONE].sum())
        update_data_istat()
        df = get_df_comuni()
        self.assertNotEqual(n_population_2022, df[cfg.TAG_POPOLAZIONE].sum())

    def check_dataframe(self, df, non_empty_columns, numeric_columns=[]):
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(df.shape[0], 0)
        self.assertTrue(set(df.columns).issuperset(set(non_empty_columns)))
        for col in non_empty_columns:
            self.assertEqual(0, df[col].isna().sum(), f"Column {col} has {df[col].isna().sum()} NaN values.")
        for col in numeric_columns:
            self.assertEqual(0, (df[col] <= 0).sum())

    def test_get_df_comuni(self):
        df = get_df_comuni()
        non_empty_columns = [cfg.TAG_COMUNE, cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_POPOLAZIONE,
                             cfg.TAG_SUPERFICIE]
        self.check_dataframe(df, non_empty_columns)
        for col in non_empty_columns:
            pos = df[col].notnull()
            if pos.sum() > 0:
                self.logger.warning(f"{col} not found for comuni: {','.join(df[pos][cfg.TAG_COMUNE].values)}")
            self.assertGreater(pos.mean(), 0.1)

    def test_get_df_province(self):
        df = get_df_province()
        non_empty_columns = [cfg.TAG_PROVINCIA, cfg.TAG_REGIONE, cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]
        numeric_columns = [cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]
        self.check_dataframe(df, non_empty_columns, numeric_columns)

    def test_get_df_regioni(self):
        df = get_df_regioni()
        non_empty_columns = [cfg.TAG_REGIONE, cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]
        numeric_columns = [cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]
        self.check_dataframe(df, non_empty_columns, numeric_columns)

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
