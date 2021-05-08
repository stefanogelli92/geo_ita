import os
from datetime import datetime
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
import geopandas as gpd

from geo_ita.src.definition import *
import geo_ita.src.config as cfg


# Todo Remove multiple file in folder
# Todo Print Last date update

def _get_list():
    df = get_anagrafica_df()
    result = [list(df[cfg.TAG_COMUNE].values),
              list(df[cfg.TAG_PROVINCIA].unique()),
              list(df[cfg.TAG_SIGLA].unique()),
              list(df[cfg.TAG_REGIONE].unique())]
    return result


def get_list_comuni():
    df = get_anagrafica_df()
    result = list(df[cfg.TAG_COMUNE].values)
    return result


def get_list_province():
    df = get_anagrafica_df()
    result = list(df[cfg.TAG_PROVINCIA].unique())
    return result


def get_list_regioni():
    df = get_anagrafica_df()
    result = list(df[cfg.TAG_REGIONE].unique())
    return result


def __get_last_file_from_folder(path):
    files = os.listdir(path)
    last_files = ""
    last_date = datetime(1999, 1, 1)
    for f in files:
        try:
            date = pd.to_datetime(f.split(".")[0][-10:], format="%d_%m_%Y")
            if date > last_date:
                last_date = date
                last_files = f
        except:
            pass
    return last_files, last_date


def __get_last_shape_file_from_folder(path):
    files = os.listdir(path)
    last_files = ""
    last_date = datetime(1999, 1, 1)
    for f in files:
        if f.split(".")[1] == "shp":
            date = pd.to_datetime(f.split("_")[0][-8:], format="%d%m%Y")
            if date > last_date:
                last_date = date
                last_files = f
    return last_files, last_date


def __rename_col(df, rename_dict):
    df.columns = [rename_dict[x] if x in rename_dict else x for x in df.columns]


def get_anagrafica_df():
    """
    Returns
    Restituisce un dataset contenente un'anagrafica ISTAT dei comuni italiani con il dettaglio delle province di
    appartenenza e delle regioni.
    """
    path = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"])
    last_files, _ = __get_last_file_from_folder(path)

    df = pd.read_excel(path / PureWindowsPath(last_files))

    __rename_col(df, cfg.anagrafica_comuni["column_rename"])
    df[cfg.TAG_CODICE_COMUNE] = df[cfg.TAG_CODICE_COMUNE].astype(int)
    df[cfg.TAG_CODICE_PROVINCIA] = df[cfg.TAG_CODICE_PROVINCIA].astype(int)
    df[cfg.TAG_CODICE_REGIONE] = df[cfg.TAG_CODICE_REGIONE].astype(int)
    df[cfg.TAG_REGIONE] = df[cfg.TAG_REGIONE].str.split("/").str[0]
    df[cfg.TAG_SIGLA].fillna("NA", inplace=True)
    df[cfg.TAG_PROVINCIA] = df[cfg.TAG_PROVINCIA].str.split("/").str[0]
    return df


def create_double_languages_mapping():
    path = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"])
    last_files, _ = __get_last_file_from_folder(path)

    df = pd.read_excel(path / PureWindowsPath(last_files))
    tag_ita = "Denominazione in italiano"
    tag_2 = "Denominazione (Italiana e straniera)"
    df = df[[tag_ita, tag_2]]
    df = df[df[tag_ita] != df[tag_2]]
    sep = "&&"
    df[tag_2] = np.where(df[tag_2].str.contains("/"), df[tag_2].str.replace("/", sep), df[tag_2].str.replace("-", sep))
    df[tag_2] = df[tag_2].str.split(sep)
    df = df.explode(tag_2)
    df[tag_2] = df[tag_2].str.lower()
    df[tag_ita] = df[tag_ita].str.lower()
    df = df[df[tag_ita] != df[tag_2]]
    df = df.set_index(tag_2)[tag_ita].to_dict()
    return df


def get_variazioni_amministrative_df():
    path = root_path / PureWindowsPath(cfg.variazioni_amministrative["path"])
    last_files, _ = __get_last_file_from_folder(path)

    df = pd.read_csv(path / PureWindowsPath(last_files), encoding='latin-1', sep=";")

    __rename_col(df, cfg.variazioni_amministrative["column_rename"])
    df = df[df["tipo_variazione"].isin(["ES", "CD"])]
    df = df[~df["Contenuto del provvedimento"].str.contains("accanto alla denominazione in lingua italiana")]
    return df


def get_popolazione_df():
    """
    Returns
    Restituisce un dataset contenente la popolazione dei singoli comuni italiani (dato ISTAT)
    """
    path = root_path / PureWindowsPath(cfg.popolazione_comuni["path"])
    last_files, _ = __get_last_file_from_folder(path)
    df = pd.read_csv(path / PureWindowsPath(last_files))
    df = df[df["Territorio"] != "Italia"]
    __rename_col(df, cfg.popolazione_comuni["column_rename"])
    df[cfg.TAG_CODICE_COMUNE] = df[cfg.TAG_CODICE_COMUNE].astype(int)
    return df


def get_comuni_shape_df():
    """
    Returns
    Restituisce un dataset con le shape di ciascun comune italiano (Provenienza Istat).
    Dataset utilizzato per fare plot geografici dell'italia.
    """
    path = root_path / PureWindowsPath(cfg.shape_comuni["path"])
    last_files, _ = __get_last_shape_file_from_folder(path)
    df = gpd.read_file(path / PureWindowsPath(last_files), encoding='utf-8')
    __rename_col(df, cfg.shape_comuni["column_rename"])
    df["center_x"] = df["geometry"].centroid.x
    df["center_y"] = df["geometry"].centroid.y
    df[cfg.TAG_CODICE_COMUNE] = df[cfg.TAG_CODICE_COMUNE].astype(int)
    return df


def get_province_shape_df():
    """
    Returns
    Restituisce un dataset con le shape di ciascun comune italiano (Provenienza Istat).
    Dataset utilizzato per fare plot geografici dell'italia.
    """
    path = root_path / PureWindowsPath(cfg.shape_province["path"])
    last_files, _ = __get_last_shape_file_from_folder(path)
    df = gpd.read_file(path / PureWindowsPath(last_files), encoding='utf-8')
    __rename_col(df, cfg.shape_province["column_rename"])
    df["center_x"] = df["geometry"].centroid.x
    df["center_y"] = df["geometry"].centroid.y
    df[cfg.TAG_CODICE_PROVINCIA] = df[cfg.TAG_CODICE_PROVINCIA].astype(int)
    return df


def get_regioni_shape_df():
    """
    Returns
    Restituisce un dataset con le shape di ciascun comune italiano (Provenienza Istat).
    Dataset utilizzato per fare plot geografici dell'italia.
    """
    path = root_path / PureWindowsPath(cfg.shape_regioni["path"])
    last_files, _ = __get_last_shape_file_from_folder(path)
    df = gpd.read_file(path / PureWindowsPath(last_files), encoding='utf-8')
    __rename_col(df, cfg.shape_regioni["column_rename"])
    df["center_x"] = df["geometry"].centroid.x
    df["center_y"] = df["geometry"].centroid.y
    df[cfg.TAG_CODICE_REGIONE] = df[cfg.TAG_CODICE_REGIONE].astype(int)
    return df


def get_dimensioni_df():
    """
    Returns
    Restituisce un dataset contenente un'anagrafica ISTAT dei comuni italiani con il dettaglio delle province di
    appartenenza e delle regioni.
    """
    path = root_path / PureWindowsPath(cfg.dimensioni_comuni["path"])
    last_files, _ = __get_last_file_from_folder(path)

    df = pd.read_excel(path / PureWindowsPath(last_files), sheet_name="Dati comunali")

    __rename_col(df, cfg.dimensioni_comuni["column_rename"])
    df = df[df[cfg.TAG_CODICE_COMUNE].notnull()]
    df[cfg.TAG_CODICE_COMUNE] = df[cfg.TAG_CODICE_COMUNE].astype(int)
    return df


def get_df(level):
    if level == cfg.LEVEL_COMUNE:
        result = create_df_comuni()
    elif level == cfg.LEVEL_PROVINCIA:
        result = create_df_province()
    elif level == cfg.LEVEL_REGIONE:
        result = create_df_regioni()
    else:
        raise Exception("Unknown level")
    return result


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
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    shape = get_comuni_shape_df()[cfg.shape_comuni["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_COMUNE)
    dimensioni = get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
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
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    dimensioni = get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
    df = df.groupby([cfg.TAG_PROVINCIA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
                     cfg.TAG_REGIONE])[[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]].sum().reset_index()
    df = df.replace({'NAN': None})
    shape = get_province_shape_df()[cfg.shape_province["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_PROVINCIA)
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
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    dimensioni = get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
    df = df.groupby([cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE])[
        [cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]].sum().reset_index()
    shape = get_regioni_shape_df()[cfg.shape_regioni["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_REGIONE)
    return df
