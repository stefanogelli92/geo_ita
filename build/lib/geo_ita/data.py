import os
from geo_ita.definition import *
from datetime import datetime
import pandas as pd
import geopandas as gpd
import geo_ita.config as cfg
from pathlib import PureWindowsPath


def _get_last_file_from_folder(path):
    files = os.listdir(path)
    last_files = ""
    last_date = datetime(1999, 1, 1)
    for f in files:
        date = pd.to_datetime(f.split(".")[0][-10:], format="%d_%m_%Y")
        if date > last_date:
            last_date = date
            last_files = f
    return last_files, last_date


def _rename_col(df, rename_dict):
    df.columns = [rename_dict[x] if x in rename_dict else x for x in df.columns]


def get_anagrafica_df():
    """
    Returns
    Restituisce un dataset contenente un'anagrafica ISTAT dei comuni italiani con il dettaglio delle province di
    appartenenza e delle regioni.
    """
    path = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"])
    last_files, _ = _get_last_file_from_folder(path)

    df = pd.read_excel(path / PureWindowsPath(last_files))

    _rename_col(df, cfg.anagrafica_comuni["column_rename"])

    df["Regione"] = df["Regione"].str.split("/").str[0]
    df["sigla"].fillna("NAN", inplace=True)
    df["denominazione_provincia"] = df["denominazione_provincia"].str.split("/").str[0]
    return df


def get_popolazione_df():
    """
    Returns
    Restituisce un dataset contenente la popolazione dei singoli comuni italiani (dato ISTAT)
    """
    path = root_path / PureWindowsPath(cfg.popolazione_comuni["path"])
    last_files, _ = _get_last_file_from_folder(path)
    df = pd.read_csv(path / PureWindowsPath(last_files))
    df = df[df["Territorio"] != "Italia"]
    _rename_col(df, cfg.popolazione_comuni["column_rename"])
    return df


def get_comuni_shape_df():
    """
    Returns
    Restituisce un dataset con le shape di ciascun comune italiano (Provenienza Istat).
    Dataset utilizzato per fare plot geografici dell'italia.
    """
    path = root_path / PureWindowsPath(cfg.shape_comuni["path"])
    last_files, _ = _get_last_file_from_folder(path)
    df = gpd.read_file(path / PureWindowsPath(last_files), encoding='utf-8')
    _rename_col(df, cfg.shape_comuni["column_rename"])
    return df


def get_province_shape_df():
    """
    Returns
    Restituisce un dataset con le shape di ciascun comune italiano (Provenienza Istat).
    Dataset utilizzato per fare plot geografici dell'italia.
    """
    path = root_path / PureWindowsPath(cfg.shape_province["path"])
    last_files, _ = _get_last_file_from_folder(path)
    df = gpd.read_file(path / PureWindowsPath(last_files), encoding='utf-8')
    _rename_col(df, cfg.shape_province["column_rename"])
    return df


def get_regioni_shape_df():
    """
    Returns
    Restituisce un dataset con le shape di ciascun comune italiano (Provenienza Istat).
    Dataset utilizzato per fare plot geografici dell'italia.
    """
    path = root_path / PureWindowsPath(cfg.shape_regioni["path"])
    last_files, _ = _get_last_file_from_folder(path)
    df = gpd.read_file(path / PureWindowsPath(last_files), encoding='utf-8')
    _rename_col(df, cfg.shape_regioni["column_rename"])
    return df


def get_dimensioni_df():
    """
    Returns
    Restituisce un dataset contenente un'anagrafica ISTAT dei comuni italiani con il dettaglio delle province di
    appartenenza e delle regioni.
    """
    path = root_path / PureWindowsPath(cfg.dimensioni_comuni["path"])
    last_files, _ = _get_last_file_from_folder(path)

    df = pd.read_excel(path / PureWindowsPath(last_files), sheet_name="Dati comunali")

    _rename_col(df, cfg.dimensioni_comuni["column_rename"])

    return df


def create_df_comuni():
    """
    Returns
    Restituisce un dataset con le seguenti info a livello di comune:
     - Popolazione
     - Dimensione
     - Shape
     - provincia + Regione
    """
    anagrafica = get_anagrafica_df()[cfg.anagrafica_comuni["column_rename"].values()]
    popolazione = get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    popolazione["codice_comune"] = popolazione["codice_comune"].astype(int)
    popolazione.drop(['denominazione_comune'], axis=1, inplace=True)
    df = anagrafica.merge(popolazione, how="left", on="codice_comune")
    shape = get_comuni_shape_df()[cfg.shape_comuni["column_rename"].values()]
    shape["center_x"] = shape["geometry"].centroid.x
    shape["center_y"] = shape["geometry"].centroid.y
    df = df.merge(shape, how="left", on="codice_comune")
    dimensioni = get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    dimensioni.drop(['codice_provincia', 'codice_regione'], axis=1, inplace=True)
    df = df.merge(dimensioni, how="left", on="codice_comune")
    return df


def create_df_province():
    """
    Returns
    Restituisce un dataset con le seguenti info a livello di provincia:
     - Popolazione
     - Dimensione
     - Shape
     - Regione
    """
    anagrafica = get_anagrafica_df()[cfg.anagrafica_comuni["column_rename"].values()]
    popolazione = get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    popolazione["codice_comune"] = popolazione["codice_comune"].astype(int)
    popolazione.drop(['denominazione_comune'], axis=1, inplace=True)
    df = anagrafica.merge(popolazione, how="left", on="codice_comune")
    df["sigla"].fillna("NAN", inplace=True)
    df = df.groupby(["denominazione_provincia", "codice_provincia", "Regione", "sigla"])["popolazione"].sum().reset_index()
    df = df.replace({'NAN': None})
    shape = get_province_shape_df()[cfg.shape_province["column_rename"].values()]
    shape["center_x"] = shape["geometry"].centroid.x
    shape["center_y"] = shape["geometry"].centroid.y
    df = df.merge(shape, how="left", on="codice_provincia")
    dimensioni = get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    dimensioni = dimensioni.groupby("codice_provincia")["superficie_km2"].sum().reset_index()
    df = df.merge(dimensioni, how="left", on="codice_provincia")
    return df


def create_df_regioni():
    """
    Returns
    Restituisce un dataset con le seguenti info a livello di regione:
     - Popolazione
     - Dimensione
     - Shape
    """
    anagrafica = get_anagrafica_df()[cfg.anagrafica_comuni["column_rename"].values()]
    popolazione = get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    popolazione["codice_comune"] = popolazione["codice_comune"].astype(int)
    popolazione.drop(['denominazione_comune'], axis=1, inplace=True)
    df = anagrafica.merge(popolazione, how="left", on="codice_comune")
    df = df.groupby(["codice_regione", "Regione"])["popolazione"].sum().reset_index()
    shape = get_regioni_shape_df()[cfg.shape_regioni["column_rename"].values()]
    shape["center_x"] = shape["geometry"].centroid.x
    shape["center_y"] = shape["geometry"].centroid.y
    df = df.merge(shape, how="left", on="codice_regione")
    dimensioni = get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    dimensioni = dimensioni.groupby("codice_regione")["superficie_km2"].sum().reset_index()
    df = df.merge(dimensioni, how="left", on="codice_regione")
    return df


get_popolazione_df()