import re
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd
import unidecode

import geo_ita.src.config as cfg


class GeoLevel(Enum):
    COORDINATES = "coordinates"
    COMUNE = "comune"
    PROVINCIA = "provincia"
    REGIONE = "regione"
    COUNTRY = "country"

    # Define less than to allow sorting
    def __lt__(self, other):
        if not isinstance(other, GeoLevel):
            return NotImplemented
        # Compare based on their order in the Enum
        return list(GeoLevel).index(self) < list(GeoLevel).index(other)

    # The sum will concatenate the name of the value
    def __str__(self):
        return self.value

    def __radd__(self, other):
        if isinstance(other, str):
            return other + str(self)
        return NotImplemented


simplify_values = {GeoLevel.REGIONE: 500,
                   GeoLevel.PROVINCIA: 500,
                   GeoLevel.COMUNE: 250}


class CodeLevel(Enum):
    CODE = "code"
    SIGLA = "sigla"
    DENOMINATION = "denomination"
    COORDINATES = "coordinates"


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


def get_tag_registry(code, level):
    if level == GeoLevel.COMUNE:
        if code == CodeLevel.CODE:
            result = cfg.TAG_CODICE_COMUNE
        else:
            result = cfg.TAG_COMUNE
    elif level == GeoLevel.PROVINCIA:
        if code == CodeLevel.CODE:
            result = cfg.TAG_CODICE_PROVINCIA
        elif code == CodeLevel.SIGLA:
            result = cfg.TAG_SIGLA
        else:
            result = cfg.TAG_PROVINCIA
    elif level == GeoLevel.REGIONE:
        if code == CodeLevel.CODE:
            result = cfg.TAG_CODICE_REGIONE
        else:
            result = cfg.TAG_REGIONE
    elif level == GeoLevel.COUNTRY:
        if code == CodeLevel.DENOMINATION:
            result = cfg.TAG_COUNTRY
        else:
            raise Exception("Only denomination for country.")
    elif level == GeoLevel.COORDINATES:
        if code == CodeLevel.DENOMINATION:
            result = cfg.TAG_COORDINATES
        else:
            raise Exception("Only denomination for country.")
    else:
        raise Exception("Level UNKNOWN")
    return result


def clean_htmltext(text: str) -> str:
    text = text.lower()
    text = re.sub('[^A-Za-z0-9.]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(' +', ' ', text)
    return text


def test_column_in_dataframe(df: pd.DataFrame, column):
    if column not in df.columns:
        raise Exception(f"Column {column} not found in DataFrame.")


def clean_denomination_text_value(value):
    value = value.lower()  # All strig in lowercase
    value = re.sub(r'[^\w\s]', ' ', value)  # Remove non alphabetic characters
    value = value.strip()
    value = re.sub(r'\s+', ' ', value)
    value = cfg.comuni_exceptions.get(value, value)
    value = unidecode.unidecode(value)
    value = cfg.comuni_exceptions.get(value, value)
    for k, v in cfg.clear_denomination.items():
        value = value.replace(k, v)
    return value