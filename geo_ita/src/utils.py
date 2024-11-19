from enum import Enum
from typing import List, Union

import numpy as np
import geo_ita.src.config as cfg


class GeoLevel(Enum):
    COMUNE = "comune"
    PROVINCIA = "provincia"
    REGIONE = "regione"

    # Define less than to allow sorting
    def __lt__(self, other):
        if not isinstance(other, GeoLevel):
            return NotImplemented
        # Compare based on their order in the Enum
        return list(GeoLevel).index(self) < list(GeoLevel).index(other)


simplify_values = {GeoLevel.REGIONE: 500,
                   GeoLevel.PROVINCIA: 500,
                   GeoLevel.COMUNE: 250}


class CodeLevel(Enum):
    CODE = "code"
    SIGLA = "sigla"
    DENOMINATION = "denomination"


def infer_geographical_category(list_values: List[Union[str, int, float]]) -> CodeLevel:
    """
    Determine if the list of values represents codes (ISTAT), comuni, provincie o regioni names, or abbreviations (sigle).

    Parameters:
        list_values (List[Union[str, int, float]]): List of values to analyze.

    Returns:
        str: The determined code, either CODE_DENOMINAZIONE, CODE_CODICE_ISTAT, or CODE_SIGLA.
    """
    # Filter out NaN and None values
    list_values = [x for x in list_values if str(x) != 'nan' and x is not None]

    n_tot = len(list_values)

    if n_tot == 0:
        return CodeLevel.DENOMINATION

    # Check if the majority of values are numeric (potentially ISTAT codes)
    code_count = sum((isinstance(item, (int, float, np.integer, np.floating)) or
                      (isinstance(item, str) and item.isdigit())) for item in list_values)

    if code_count / n_tot > 0.8:
        return CodeLevel.CODE

    # Check if the majority of values are 2-letter abbreviations
    sigla_count = sum(isinstance(item, str) and item.isalpha() and len(item) == 2 for item in list_values)
    if sigla_count / n_tot > 0.8:
        return CodeLevel.SIGLA

    # Default to denominations if no other condition is met
    return CodeLevel.DENOMINATION

