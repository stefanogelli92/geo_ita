import re
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd
import unidecode
from matplotlib.colors import LinearSegmentedColormap

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
        return list(GeoLevel).index(self) < list(GeoLevel).index(other)

    def __le__(self, other):
        if not isinstance(other, GeoLevel):
            return NotImplemented
        return list(GeoLevel).index(self) <= list(GeoLevel).index(other)

    def __gt__(self, other):
        if not isinstance(other, GeoLevel):
            return NotImplemented
        return list(GeoLevel).index(self) > list(GeoLevel).index(other)

    def __ge__(self, other):
        if not isinstance(other, GeoLevel):
            return NotImplemented
        return list(GeoLevel).index(self) >= list(GeoLevel).index(other)

    # The sum will concatenate the name of the value
    def __str__(self):
        return self.value

    def __radd__(self, other):
        if isinstance(other, str):
            return other + str(self)
        return NotImplemented


class CodeLevel(Enum):
    CODE = "code"
    SIGLA = "sigla"
    DENOMINATION = "denomination"
    COORDINATES = "coordinates"


class Check(Enum):
    OK = "OK"
    WARNING = "Warning"
    SOLVED = "Warning solved"


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
    tag_mapping = {
        GeoLevel.COMUNE: {
            CodeLevel.CODE: cfg.TAG_CODICE_COMUNE,
            CodeLevel.DENOMINATION: cfg.TAG_COMUNE
        },
        GeoLevel.PROVINCIA: {
            CodeLevel.CODE: cfg.TAG_CODICE_PROVINCIA,
            CodeLevel.SIGLA: cfg.TAG_SIGLA,
            CodeLevel.DENOMINATION: cfg.TAG_PROVINCIA
        },
        GeoLevel.REGIONE: {
            CodeLevel.CODE: cfg.TAG_CODICE_REGIONE,
            CodeLevel.DENOMINATION: cfg.TAG_REGIONE
        },
        GeoLevel.COUNTRY: {
            CodeLevel.DENOMINATION: cfg.TAG_COUNTRY
        },
        GeoLevel.COORDINATES: {
            CodeLevel.DENOMINATION: cfg.TAG_COORDINATES
        }
    }

    if level not in tag_mapping:
        raise Exception("Level UNKNOWN")

    if code not in tag_mapping[level]:
        raise Exception("Invalid code for the given level")

    return tag_mapping[level][code]


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


def ensure_list(value, default=None):
    if value is None:
        return default
    if not isinstance(value, list):
        value = [value]
    return value


def _human_format(num):
    # Show float number in more readable format
    num = float('{:.2g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def _linear_colormap(color_name1="white", color_name2=None, minval=0, maxval=1):
    # Create a 2 color linear map
    if color_name2 is None:
        color_name2 = "blue"
    cmap = _truncate_colormap(LinearSegmentedColormap.from_list("", [color_name1, color_name2]), minval=minval,
                              maxval=maxval)
    return cmap


def check_duplicate_column_output(original_df, output_df, output_columns, suffix, handle_duplicate_column, log):
    if suffix is not None:
        rename_columns = {
            col: col + suffix
            for col in output_columns
        }
        output_df.rename(columns=rename_columns, inplace=True)
        output_columns = [col + suffix for col in output_columns]
    column_duplicates = list(set(output_columns).intersection(original_df.columns))
    if len(column_duplicates) > 0:
        if handle_duplicate_column == "error":
            raise Exception(f"Found column in dataset with the same name of one of the output columns:\n"
                            f"{column_duplicates}, change the 'handle_duplicate_column' params in order to handle "
                            f"those columns.\n'overwrite'= the original columns will be overwrite.\n"
                            f"suffix (str): this string will be used as suffix for the new columns.")
        elif handle_duplicate_column == "overwrite":
            log.warning(f"Columns: {column_duplicates} will be overwrite in dataset.")
            original_df.drop(columns=column_duplicates, inplace=True)
        elif handle_duplicate_column == "progressive":
            log.warning(f"Found column in dataset with the same name of one of the output columns:\n"
                        f"{column_duplicates}, the new columns will be added with a progressive suffix.")
            for col in column_duplicates:
                i = 1
                while f"{col}_{i}" in original_df.columns:
                    i += 1
                output_df.rename(columns={col: f"{col}_{i}"}, inplace=True)
        else:
            raise Exception(f"Invalid value for 'handle_duplicate_column' params: {handle_duplicate_column}.")

