import math
import os
from datetime import datetime
from pathlib import PureWindowsPath
import logging
import requests

import numpy as np
import pandas as pd
import geopandas as gpd
import urllib
import zipfile
from valdec.decorators import validate


from geo_ita.src.definition import *
import geo_ita.src.config as cfg

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def _get_list():
    df = get_df_comuni()
    result = [list(df[cfg.TAG_COMUNE].values),
              list(df[cfg.TAG_PROVINCIA].unique()),
              list(df[cfg.TAG_SIGLA].unique()),
              list(df[cfg.TAG_REGIONE].unique())]
    return result


def get_list_comuni():
    df = get_df_comuni()
    result = list(df[cfg.TAG_COMUNE].values)
    return result


def get_list_province():
    df = get_df_province()
    result = list(df[cfg.TAG_PROVINCIA].unique())
    return result


def get_list_regioni():
    df = get_df_regioni()
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


def _get_anagrafica_df():
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


def get_double_languages_mapping():
    df = get_df_comuni()
    tag_ita = cfg.TAG_COMUNE
    tag_2 = "denominazione_comune_ita_straniera"
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


def create_variazioni_amministrative_df():
    path = root_path / PureWindowsPath(cfg.variazioni_amministrative["path"])
    last_files, _ = __get_last_file_from_folder(path)

    df = pd.read_csv(path / PureWindowsPath(last_files), encoding='latin-1') #, sep=";"

    __rename_col(df, cfg.variazioni_amministrative["column_rename"])
    df = df[df["tipo_variazione"].isin(["ES", "CD"])]
    df["Contenuto del provvedimento"].fillna("", inplace=True)
    df = df[~df["Contenuto del provvedimento"].str.contains("accanto alla denominazione in lingua italiana")]
    df.to_pickle(root_path / PureWindowsPath(cfg.df_variazioni_mapping["path"]))
    return


def get_variazioni_amministrative_df():
    return pd.read_pickle(root_path / PureWindowsPath(cfg.df_variazioni_mapping["path"]))


def _get_popolazione_df():
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


def _get_comuni_shape_df():
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


def _get_province_shape_df():
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


def _get_regioni_shape_df():
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


def _get_dimensioni_df():
    """
    Returns
    Restituisce un dataset contenente un'anagrafica ISTAT dei comuni italiani con il dettaglio delle province di
    appartenenza e delle regioni.
    """
    path = root_path / PureWindowsPath(cfg.dimensioni_comuni["path"])
    #last_files, _ = __get_last_file_from_folder(path)

    df = pd.read_pickle(path / PureWindowsPath(path))

    __rename_col(df, cfg.dimensioni_comuni["column_rename"])
    df = df[df[cfg.TAG_CODICE_COMUNE].notnull()]
    df[cfg.TAG_CODICE_COMUNE] = df[cfg.TAG_CODICE_COMUNE].astype(int)
    return df


def get_df(level):
    if level == cfg.LEVEL_COMUNE:
        result = get_df_comuni()
    elif level == cfg.LEVEL_PROVINCIA:
        result = get_df_province()
    elif level == cfg.LEVEL_REGIONE:
        result = get_df_regioni()
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
    anagrafica = _get_anagrafica_df()[cfg.anagrafica_comuni["column_rename"].values()]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    shape = _get_comuni_shape_df()[cfg.shape_comuni["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_COMUNE)
    dimensioni = _get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
    df.to_pickle(root_path / PureWindowsPath(cfg.df_comuni["path"]))
    return


def get_df_comuni():
    return pd.read_pickle(root_path / PureWindowsPath(cfg.df_comuni["path"]))


def create_df_province():
    """
    Returns
    Restituisce un dataset con le seguenti info a livello di provincia:
     - Popolazione
     - Dimensione
     - Shape
     - Regione
    """
    anagrafica = _get_anagrafica_df()[cfg.anagrafica_comuni["column_rename"].values()]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    dimensioni = _get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
    df = df.groupby([cfg.TAG_PROVINCIA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
                     cfg.TAG_REGIONE, cfg.TAG_AREA_GEOGRAFICA])[[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]].sum().reset_index()
    df = df.replace({'NAN': None})
    shape = _get_province_shape_df()[cfg.shape_province["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_PROVINCIA)
    df.to_pickle(root_path / PureWindowsPath(cfg.df_province["path"]))
    return


def get_df_province():
    return pd.read_pickle(root_path / PureWindowsPath(cfg.df_province["path"]))


def create_df_regioni():
    """
    Returns
    Restituisce un dataset con le seguenti info a livello di regione:
     - Popolazione
     - Dimensione
     - Shape
    """
    anagrafica = _get_anagrafica_df()[cfg.anagrafica_comuni["column_rename"].values()]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    dimensioni = _get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
    df = df.groupby([cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE, cfg.TAG_AREA_GEOGRAFICA])[
        [cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]].sum().reset_index()
    shape = _get_regioni_shape_df()[cfg.shape_regioni["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_REGIONE)
    df.to_pickle(root_path / PureWindowsPath(cfg.df_regioni["path"]))
    return df


def get_df_regioni():
    return pd.read_pickle(root_path / PureWindowsPath(cfg.df_regioni["path"]))


def _get_shape_italia():
    df = get_df_regioni()
    df["key"] = "Italia"
    df = gpd.GeoDataFrame(df, geometry="geometry")
    df = df.dissolve(by='key')
    return df


def _download_high_resolution_population_density_df():
    link = cfg.high_resolution_population_density["link"]
    file_name = link.split("/")[-1]
    folder_path = root_path / PureWindowsPath(cfg.high_resolution_population_density["folder_path"])
    file_path = PureWindowsPath(folder_path) / PureWindowsPath(file_name)
    log.info("Start downloading the high resolution population of Italy (239.0M)")
    start = datetime.now()
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(link, file_path)
    end = datetime.now()
    log.info("Dowload High resolution population of Italy ended in {}".format(end-start))
    new_file_path = PureWindowsPath(folder_path) / PureWindowsPath(file_name.replace(".zip", ""))
    log.info("Start unzipping the file")
    start = datetime.now()
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    os.remove(file_path)
    end = datetime.now()
    log.info("Unzipping ended in {}".format(end-start))
    df = pd.read_csv(new_file_path)

    # Reduce memory usage
    df["Lat"] = df["Lat"].astype('float32')
    df["Lon"] = df["Lon"].astype('float32')
    df["Population"] = (df["Population"]*cfg.high_resolution_population_density["moltiplicative_factor"]).round(0).astype('int32')
    df = df[df["Population"] > 0]
    df.to_pickle(root_path / PureWindowsPath(cfg.high_resolution_population_density["file_path"]))
    os.remove(new_file_path)
    return df


@validate
def get_high_resolution_population_density_df() -> pd.DataFrame:
    file_path = root_path / PureWindowsPath(cfg.high_resolution_population_density["file_path"])
    if os.path.exists(file_path):
        log.info("Start loading the file")
        start = datetime.now()
        df = pd.read_pickle(file_path)
        end = datetime.now()
        log.info("File loaded in {}".format(end - start))
    else:
        df = _download_high_resolution_population_density_df()
    df["Population"] = df["Population"] / cfg.high_resolution_population_density["moltiplicative_factor"]
    return df


def remove_high_resolution_population_density_file():
    file_path = root_path / PureWindowsPath(cfg.high_resolution_population_density["file_path"])
    if os.path.isfile(file_path):
        os.remove(file_path)


def create_df():
    create_df_comuni()
    create_df_province()
    create_df_regioni()
    create_variazioni_amministrative_df()


def upload_data_istat_from_api():
    year = 2020
    path1 = root_path / PureWindowsPath(cfg.dimensioni_comuni["path"].replace("pkl", "csv"))
    path2 = root_path / PureWindowsPath(cfg.dimensioni_comuni["path"])
    base_url = "http://sdmx.istat.it/SDMXWS/rest/data/729_1050/A..TOTAREA2?startPeriod={year}".format(year=year)
    response = requests.get(base_url, headers={'Accept': 'application/vnd.sdmx.data+csv;version=1.0.0'}, verify=False)
    url_content = response.content
    csv_file = open(path1, 'wb')
    csv_file.write(url_content)
    csv_file.close()
    df = pd.read_csv(path1)
    df = df[(df["TIME_PERIOD"] == year) &
            (df["ITTER107"].astype(str).str.isnumeric())]
    df = df[["ITTER107", "OBS_VALUE"]]
    df.columns = ["Codice Comune", "Superficie totale (Km2)"]
    df.to_pickle(path2)
    os.remove(path1)
