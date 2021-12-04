from geo_ita.src._data import *
from geo_ita.src.definition import *

import unittest
import logging
from pathlib import PureWindowsPath


class TestEnrichDataframe(unittest.TestCase):

    def test_download_high_density_population_df(self):
        remove_high_resolution_population_density_file()
        df = get_high_resolution_population_density_df()
        del df
        df = get_high_resolution_population_density_df()


def test_run():
    prova_1 = get_df_comuni()
    prova_2 = get_df_province()
    prova_3 = get_df_regioni()
    prova_4 = get_list_comuni()
    prova_5 = get_list_province()
    prova_6 = get_list_regioni()

    prova = "Debug"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_run()
    unittest.main()
