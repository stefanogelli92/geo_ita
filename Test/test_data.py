from geo_ita.data import *
from geo_ita.src._data import _download_high_density_population_df
from geo_ita.src._enrich_dataframe import _process_high_density_population_df
from geo_ita.src.definition import *

import unittest
import logging
from pathlib import PureWindowsPath


class TestEnrichDataframe(unittest.TestCase):

    def test_download_high_density_population_df(self):
        path = root_path / PureWindowsPath("data_sources/Test")
        file_name = _download_high_density_population_df(path)
        _process_high_density_population_df(file_name)



def test_run():
    prova_1 = get_df_comuni()
    print(get_df_comuni()[["denominazione_comune", "center_x", "center_y", "popolazione"]])
    prova_2 = get_df_province()
    prova_3 = get_df_regioni()
    prova_4 = get_list_comuni()
    prova_5 = get_list_province()
    prova_6 = get_list_regioni()

    prova = "Debug"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #test_run()
    unittest.main()