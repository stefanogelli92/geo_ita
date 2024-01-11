import math
import os
import shutil
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


def _get_list(df=None):
    if df is None:
        df = get_df_comuni()
    result = []
    if cfg.TAG_COMUNE in df.columns:
        result.append(list(df[cfg.TAG_COMUNE].values))
    else:
        result.append(None)
    if cfg.TAG_PROVINCIA in df.columns:
        result.append(list(df[cfg.TAG_PROVINCIA].values))
    else:
        result.append(None)
    if cfg.TAG_SIGLA in df.columns:
        result.append(list(df[cfg.TAG_SIGLA].values))
    else:
        result.append(None)
    if cfg.TAG_REGIONE in df.columns:
        result.append(list(df[cfg.TAG_REGIONE].values))
    else:
        result.append(None)
    return result


def get_list_comuni():
    """
        Returns
        The list of names of italian comuni.
    """
    df = get_df_comuni()
    result = list(df[cfg.TAG_COMUNE].values)
    return result


def get_list_province():
    """
        Returns
        The list of names of italian province.
    """
    df = get_df_province()
    result = list(df[cfg.TAG_PROVINCIA].unique())
    return result


def get_list_regioni():
    """
        Returns
        The list of names of italian regioni.
    """
    df = get_df_regioni()
    result = list(df[cfg.TAG_REGIONE].unique())
    return result


def __get_last_file_from_folder(path, date_format="%d_%m_%Y"):
    files = os.listdir(path)
    last_files = ""
    last_date = datetime(1999, 1, 1)
    for f in files:
        try:
            date = pd.to_datetime(f.split(".")[0][-10:], format=date_format)
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
    A Dataframe containing the ISTAT registry of italian comuni with details of the corresponding provincia and regione.
    """
    path = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"])
    df = pd.read_pickle(path)

    __rename_col(df, cfg.anagrafica_comuni["column_rename"])
    df[cfg.TAG_CODICE_COMUNE] = df[cfg.TAG_CODICE_COMUNE].astype(int)
    df[cfg.TAG_CODICE_PROVINCIA] = df[cfg.TAG_CODICE_PROVINCIA].astype(int)
    df[cfg.TAG_CODICE_REGIONE] = df[cfg.TAG_CODICE_REGIONE].astype(int)
    df[cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA] = df[cfg.TAG_REGIONE]
    df[cfg.TAG_REGIONE] = df[cfg.TAG_REGIONE].str.split("/").str[0]
    df[cfg.TAG_SIGLA].fillna("NA", inplace=True)
    df[cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA] = df[cfg.TAG_PROVINCIA]
    df[cfg.TAG_PROVINCIA] = df[cfg.TAG_PROVINCIA].str.split("/").str[0]
    return df


def _clean_denom_text(series):
    series = series.fillna("")
    series = series.astype(str)
    series = series.where(series != "", None)
    series = series.str.lower()  # All strig in lowercase
    series = series.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non alphabetic characters
    series = series.str.strip()
    series = series.str.replace(r'\s+', ' ', regex=True)
    series = series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')  # Remove accent
    return series


def get_double_languages_mapping_comuni(df=None):
    if df is None:
        df = get_df_comuni()
    tag_ita = cfg.TAG_COMUNE
    tag_2 = cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA
    df = df[[tag_ita, tag_2]]
    df = df[df[tag_ita] != df[tag_2]]
    sep = "&&"
    df[tag_2] = np.where(df[tag_2].str.contains("/"), df[tag_2].str.replace("/", sep), df[tag_2].str.replace("-", sep))
    df[tag_2] = df[tag_2].str.split(sep)
    df = df.explode(tag_2)
    df[tag_2] = _clean_denom_text(df[tag_2])
    df[tag_ita] = _clean_denom_text(df[tag_ita])
    df = df[df[tag_ita] != df[tag_2]]
    df = df.set_index(tag_2)[tag_ita].to_dict()
    return df


def get_double_languages_mapping_province(df=None):
    if df is None:
        df = get_df_province()
    tag_ita = cfg.TAG_PROVINCIA
    tag_2 = cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA
    df = df[[tag_ita, tag_2]]
    df = df[df[tag_ita] != df[tag_2]]
    sep = "&&"
    df[tag_2] = np.where(df[tag_2].str.contains("/"), df[tag_2].str.replace("/", sep), df[tag_2].str.replace("-", sep))
    df[tag_2] = df[tag_2].str.split(sep)
    df = df.explode(tag_2)
    df[tag_2] = _clean_denom_text(df[tag_2])
    df[tag_ita] = _clean_denom_text(df[tag_ita])
    df = df[df[tag_ita] != df[tag_2]]
    df = df.set_index(tag_2)[tag_ita].to_dict()
    return df


def get_double_languages_mapping_regioni(df=None):
    if df is None:
        df = get_df_regioni()
    tag_ita = cfg.TAG_REGIONE
    tag_2 = cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA
    df = df[[tag_ita, tag_2]]
    df = df[df[tag_ita] != df[tag_2]]
    sep = "&&"
    df[tag_2] = np.where(df[tag_2].str.contains("/"), df[tag_2].str.replace("/", sep), df[tag_2])
    df[tag_2] = df[tag_2].str.split(sep)
    df = df.explode(tag_2)
    df[tag_2] = _clean_denom_text(df[tag_2])
    df[tag_ita] = _clean_denom_text(df[tag_ita])
    df = df[df[tag_ita] != df[tag_2]]
    df = df.set_index(tag_2)[tag_ita].to_dict()
    return df


def create_administrative_changes_df():
    path = root_path / PureWindowsPath(cfg.variazioni_amministrative["path"])
    df = pd.read_pickle(path)
    __rename_col(df, cfg.variazioni_amministrative["column_rename"])
    df = df[df["tipo_variazione"].isin(["ES", "CD"])]
    df["Contenuto del provvedimento"].fillna("", inplace=True)
    df = df[~df["Contenuto del provvedimento"].str.contains("accanto alla denominazione in lingua italiana")]
    df.to_pickle(root_path / PureWindowsPath(cfg.df_variazioni_mapping["path"]))
    return


def get_administrative_changes_df():
    return pd.read_pickle(root_path / PureWindowsPath(cfg.df_variazioni_mapping["path"]))


def _get_popolazione_df():
    """
    Returns
    Restituisce un dataset contenente la popolazione dei singoli comuni italiani (dato ISTAT)
    """
    path = root_path / PureWindowsPath(cfg.popolazione_comuni["path"])
    df = pd.read_pickle(path / PureWindowsPath(path))
    __rename_col(df, cfg.popolazione_comuni["column_rename"])
    df = df[df[cfg.TAG_CODICE_COMUNE].notnull()]
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
    """
        Returns
        A dataframe with all the following details for each comune:
         - Population
         - Area
         - Shape
         - provincia and Regione
    """
    return pd.read_pickle(root_path / PureWindowsPath(cfg.df_comuni["path"]))


def create_df_province():
    anagrafica = _get_anagrafica_df()[list(cfg.anagrafica_comuni["column_rename"].values()) + [cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA]]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    dimensioni = _get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
    df = df.groupby([cfg.TAG_PROVINCIA, cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
                     cfg.TAG_REGIONE, cfg.TAG_AREA_GEOGRAFICA])[[cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]].sum().reset_index()
    df = df.replace({'NAN': None})
    shape = _get_province_shape_df()[cfg.shape_province["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_PROVINCIA)
    df.to_pickle(root_path / PureWindowsPath(cfg.df_province["path"]))
    return


def get_df_province():
    """
        Returns
        A dataframe with all the following details for each provincia:
         - Population
         - Area
         - Shape
         - Regione
    """
    return pd.read_pickle(root_path / PureWindowsPath(cfg.df_province["path"]))


def create_df_regioni():
    anagrafica = _get_anagrafica_df()[list(cfg.anagrafica_comuni["column_rename"].values()) + [cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA]]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = anagrafica.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    dimensioni = _get_dimensioni_df()[cfg.dimensioni_comuni["column_rename"].values()]
    df = df.merge(dimensioni, how="left", on=cfg.TAG_CODICE_COMUNE)
    df = df.groupby([cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE, cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA, cfg.TAG_AREA_GEOGRAFICA])[
        [cfg.TAG_POPOLAZIONE, cfg.TAG_SUPERFICIE]].sum().reset_index()
    shape = _get_regioni_shape_df()[cfg.shape_regioni["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_REGIONE)
    df.to_pickle(root_path / PureWindowsPath(cfg.df_regioni["path"]))
    return df


def get_df_regioni():
    """
        Returns
        A dataframe with all the following details for each Regione:
         - Population
         - Area
         - Shape
    """
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
    log.info(f"Dowload High resolution population of Italy ended in {end-start}")
    new_file_path = PureWindowsPath(folder_path) / PureWindowsPath(file_name.replace(".zip", "").replace("_csv", ".csv"))
    log.info("Start unzipping the file")
    start = datetime.now()
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    os.remove(file_path)
    end = datetime.now()
    log.info("Unzipping ended in {}".format(end-start))
    df = pd.read_csv(new_file_path)

    df.columns = ["Lon", "Lat", "Population"]
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
    create_administrative_changes_df()


def _update_anagrafica1():
    link = cfg.anagrafica_comuni["link"]

    path1 = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"].replace("pkl", "csv"))
    path2 = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"])

    log.info(f"Start downloading the list of comuni from link {link}.")

    urllib.request.urlretrieve(link, path1)
    df = pd.read_csv(path1, keep_default_na=False, encoding='latin-1', sep=";")
    df.to_pickle(path2)
    os.remove(path1)


def _update_anagrafica2(year):
    path1 = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"].replace("pkl", "zip"))
    first_year = 2012
    n_range = int((year - first_year) / 5)
    year0 = first_year + n_range * 5
    year1 = year0 + 4
    log.info(f"Start downloading the list of comuni ISTAT.")
    start = datetime.now()
    for i in range(3):
        try:
            link = f"https://www.istat.it/storage/codici-unita-amministrative/Archivio-elenco-comuni-codici-e-denominazioni_Anni_{year0}-{year1 - i}.zip"
            urllib.request.urlretrieve(link, path1)
            break
        except:
            pass
    end = datetime.now()
    log.info(f"DowloadFile ended in {end - start}")
    path2 = root_path / PureWindowsPath(cfg.anagrafica_comuni["path"].replace(".pkl", ""))
    log.info("Start unzipping the file")
    start = datetime.now()
    with zipfile.ZipFile(path1, 'r') as zip_ref:
        zip_ref.extractall(path2)
    os.remove(path1)
    end = datetime.now()
    log.info("Unzipping ended in {}".format(end - start))

    path3 = f"Codici-statistici-e-denominazioni-al-01_01_{year}.xls"
    for root, dirs, files in os.walk(path2):
        if path3 in files:
            path3 = os.path.join(root, path3)
            break
    df = pd.read_excel(path3, keep_default_na=False)
    df.to_pickle(root_path / PureWindowsPath(cfg.anagrafica_comuni["path"]))
    shutil.rmtree(path2)


def _update_dimension_info(ref_year, i=0, force_year=False):
    year = ref_year - i
    path1 = root_path / PureWindowsPath(cfg.dimensioni_comuni["path"].replace("pkl", "csv"))
    path2 = root_path / PureWindowsPath(cfg.dimensioni_comuni["path"])
    base_url = f"http://sdmx.istat.it/SDMXWS/rest/data/729_1050/A..TOTAREA2?startPeriod={year}"
    response = requests.get(base_url, headers={'Accept': 'application/vnd.sdmx.data+csv;version=1.0.0'}, verify=False)
    url_content = response.content
    csv_file = open(path1, 'wb')
    csv_file.write(url_content)
    csv_file.close()
    df = pd.read_csv(path1)
    if df.shape[0] == 0:
        if i == 2:
            raise Exception(f"Data on ISTAT api not found. Verify internet connection.")
        else:
            log.warning(f"Data on ISTAT api not found for year {year}.")
            if force_year:
                i = i
            else:
                i += 1
            _update_dimension_info(ref_year, i=i)
    else:
        df = df[(df["TIME_PERIOD"] == year) &
                (df["ITTER107"].astype(str).str.isnumeric())]
        df = df[["ITTER107", "OBS_VALUE"]]
        df.columns = ["Codice Comune", "Superficie totale (Km2)"]
        df.to_pickle(path2)
        os.remove(path1)


def _update_population_info(ref_year, i=0, force_year=False):
    year = ref_year - i
    path1 = root_path / PureWindowsPath(cfg.popolazione_comuni["path"].replace("pkl", "csv"))
    path2 = root_path / PureWindowsPath(cfg.popolazione_comuni["path"])
    base_url = f"http://sdmx.istat.it/SDMXWS/rest/data/22_289/A.TOTAL..9.99..?startPeriod={year}"
    response = requests.get(base_url, headers={'Accept': 'application/vnd.sdmx.data+csv;version=1.0.0'}, verify=False)
    url_content = response.content
    csv_file = open(path1, 'wb')
    csv_file.write(url_content)
    csv_file.close()
    df = pd.read_csv(path1)
    if "TIME_PERIOD" not in df.columns:
        df["TIME_PERIOD"] = None
    if (df.shape[0] == 0) | ((df["TIME_PERIOD"] == year).sum() == 0):
        if i == 2:
            raise Exception(f"Data on ISTAT api not found. Verify internet conenction.")
        else:
            log.warning(f"Data on ISTAT api not found for year {year}.")
            if force_year:
                i = i
            else:
                i += 1
            _update_population_info(ref_year, i=i)
    else:
        df = df[(df["TIME_PERIOD"] == year) &
                (df["ITTER107"].astype(str).str.isnumeric())]
        df = df[["ITTER107", "OBS_VALUE"]]
        df.columns = ["Codice Comune", "Popolazione"]
        df.to_pickle(path2)
        os.remove(path1)


def _update_shape_comuni(year, i=0, force_year=False):
    links = [
        f"{cfg.shape_comuni['link']}0101{year}.zip",
        f"https://www.istat.it/storage/cartografia/confini_amministrativi/non_generalizzati/{year}/Limiti0101{year}.zip"
    ]
    link = links[i]
    file_name = link.split("/")[-1]
    folder_path = (root_path / PureWindowsPath(cfg.shape_comuni["path"])).parent
    file_path = PureWindowsPath(folder_path) / PureWindowsPath(file_name)
    log.info(f"Start downloading the Shape File (63.4M) for the year {year} from link {link}.")
    start = datetime.now()
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(link, file_path)
    except:
        if i < len(links)-1:
            _update_shape_comuni(year, i=i+1, force_year=force_year)
            return
        elif not force_year:
            log.info(f"Link for update ISTAT shape file not found (link:{link}). \n"
                     f"ISTAT may hasn't published shape file for {year} yet, try to get file for year {year-1}.")
            _update_shape_comuni(year - 1, i=0, force_year=True)
            return
        else:
            raise Exception(f"Link for update ISTAT shape file not found (link:{link}). \n"
                            f"Verify internet conenction or ISTAT hasn't published shape file for {year} yet.")
    end = datetime.now()
    log.info(f"Dowload Shape File ended in {end - start}")
    new_file_path = PureWindowsPath(folder_path) / PureWindowsPath(
        file_name.replace(".zip", ""))
    log.info("Start unzipping the file")
    start = datetime.now()
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    os.remove(file_path)
    end = datetime.now()
    log.info("Unzipping ended in {}".format(end - start))

    # Move Files
    for _folder in os.listdir(new_file_path):
        if _folder.startswith("Com"):
            _folder_path = PureWindowsPath(new_file_path) / PureWindowsPath(_folder)
            shutil.rmtree(root_path / cfg.shape_comuni["path"])
            os.replace(_folder_path, root_path / cfg.shape_comuni["path"])
        elif _folder.startswith("Prov"):
            _folder_path = PureWindowsPath(new_file_path) / PureWindowsPath(_folder)
            shutil.rmtree(root_path / cfg.shape_province["path"])
            os.replace(_folder_path, root_path / cfg.shape_province["path"])
        elif _folder.startswith("Reg"):
            _folder_path = PureWindowsPath(new_file_path) / PureWindowsPath(_folder)
            shutil.rmtree(root_path / cfg.shape_regioni["path"])
            os.replace(_folder_path, root_path / cfg.shape_regioni["path"])

    shutil.rmtree(new_file_path)
    return


def _update_administrative_changes(year):
    path1 = root_path / PureWindowsPath(cfg.variazioni_amministrative["path"].replace("pkl", "zip"))
    link = cfg.variazioni_amministrative["link"]
    urllib.request.urlretrieve(link, path1)
    path2 = root_path / PureWindowsPath(cfg.variazioni_amministrative["path"].replace(".pkl", ""))
    log.info("Start unzipping the file")
    start = datetime.now()
    with zipfile.ZipFile(path1, 'r') as zip_ref:
        zip_ref.extractall(path2)
    os.remove(path1)
    end = datetime.now()
    log.info("Unzipping ended in {}".format(end - start))

    for root, dirs, files in os.walk(path2):
        for f in files:
            if r".csv" in f:
                path3 = os.path.join(root, f)
                break
    df = pd.read_csv(path3, encoding='latin-1', sep=";")
    if year is not None:
        df = df[df["Anno"] <= year]
    df.to_pickle(root_path / PureWindowsPath(cfg.variazioni_amministrative["path"]))
    shutil.rmtree(path2)


def update_data_istat(year=None):
    if year is None:
        year = datetime.now().year
        force_year = False
        _update_anagrafica1()
    else:
        force_year = True
        _update_anagrafica2(year)
    _update_shape_comuni(year, force_year=force_year)
    _update_dimension_info(year, force_year=force_year)
    _update_population_info(year, force_year=force_year)
    _update_administrative_changes(year)
    create_df()


