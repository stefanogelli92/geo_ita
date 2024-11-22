import os
import shutil
from datetime import datetime
import logging
import geopandas as gpd
import pandasdmx as sdmx
import urllib.request
import zipfile

from valdec.decorators import validate
from geo_ita.src.definition import *
from geo_ita.src.utils import *

# Initialize logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# Helper functions to fetch lists of Italian geographical entities

def get_comuni_list() -> list[str]:
    """
    Returns a list of names of Italian comuni (municipalities).
    """
    df = get_df_comuni()
    return list(df[cfg.TAG_COMUNE].values)


def get_province_list() -> list[str]:
    """
    Returns a list of names of Italian provinces.
    """
    df = get_df_province()
    return list(df[cfg.TAG_PROVINCIA].unique())


def get_regioni_list() -> list[str]:
    """
    Returns a list of names of Italian regions.
    """
    df = get_df_regioni()
    return list(df[cfg.TAG_REGIONE].unique())


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


def __rename_columns(df, rename_dict):
    """
    Rename columns in a dataframe based on a dictionary.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns will be renamed.
    - rename_dict (dict): A dictionary where keys are original column names and values are new names.

    Returns:
    None: Modifies the DataFrame in place.
    """
    df.columns = [rename_dict.get(x, x) for x in df.columns]


def _get_registry_df() -> pd.DataFrame:
    """
    Load and process the ISTAT registry DataFrame for Italian municipalities (comuni),
    adding provincial and regional details.

    Returns:
    pd.DataFrame: A processed DataFrame with standardized columns for Italian municipal data.
    """
    # Define the file path for registry data
    path = root_path / Path(cfg.registry_comuni["path"])

    # Load data and check for successful loading
    try:
        df = pd.read_pickle(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at the specified path: {path}")

    # Rename columns based on the provided dictionary
    __rename_columns(df, cfg.registry_comuni["column_rename"])

    # Ensure specific columns have integer types, catching errors
    for tag in [cfg.TAG_CODICE_COMUNE, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_CODICE_REGIONE]:
        if tag in df.columns:
            df[tag] = df[tag].astype(int, errors='raise')
        else:
            raise KeyError(f"Expected column '{tag}' not found in DataFrame.")

    # Duplicate region and province columns to add variations with Italian/foreign labels
    df[cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA] = df[cfg.TAG_REGIONE]
    df[cfg.TAG_REGIONE] = df[cfg.TAG_REGIONE].str.split("/").str[0]

    df[cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA] = df[cfg.TAG_PROVINCIA]
    df[cfg.TAG_PROVINCIA] = df[cfg.TAG_PROVINCIA].str.split("/").str[0]

    # Handle cases where the "sigla" (abbreviation) for Napoli is missing
    napoli_condition = (df[cfg.TAG_COMUNE].str.lower() == "napoli") & (df[cfg.TAG_SIGLA].isna())
    if napoli_condition.sum() > 0:
        df.loc[napoli_condition, cfg.TAG_SIGLA] = "NA"

    return df


def _clean_denomination_text(series):
    """
    Clean and standardize text in a pandas Series.

    Parameters:
    - series (pd.Series): The Series to be cleaned.

    Returns:
    - pd.Series: The cleaned Series.
    """
    series = series.fillna("").astype(str)
    series = series.where(series != "", None)
    series = series.str.lower()
    series = series.str.replace(r'[^\w\s]', ' ', regex=True)
    series = series.str.strip()
    series = series.str.replace(r'\s+', ' ', regex=True)
    series = series.replace(cfg.comuni_exceptions)
    series = series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    series = series.replace(cfg.comuni_exceptions)

    for old, new in cfg.clear_denomination.items():
        series = series.str.replace(old, new)

    return series


def _get_double_language_mapping(df0, tag_ita, tag_foreign):
    """
    Create a mapping dictionary for names in two languages by cleaning and standardizing variations.

    Parameters:
    - df (pd.DataFrame): DataFrame containing Italian and foreign language names.
    - tag_ita (str): Column name for the Italian denomination.
    - tag_foreign (str): Column name for the foreign language denomination.

    Returns:
    dict: A dictionary where keys are foreign language names, and values are Italian names.
    """
    # Filter only the columns needed and keep rows with differing names in both languages
    df = df0[[tag_ita, tag_foreign]].drop_duplicates()
    df = df[df[tag_ita] != df[tag_foreign]]

    # Replace delimiters and split the foreign language names
    df[tag_foreign] = df[tag_foreign].where(df[tag_foreign].str.contains(r'^(\w+)\/(\1)$', regex=True),
                                            df[tag_foreign].str.split("/"))

    # Expand rows for each alternative name in the foreign language column
    df = df.explode(tag_foreign)

    # Remove any remaining identical pairs and create the dictionary mapping
    df = df[df[tag_ita] != df[tag_foreign]]
    df = df[~df[tag_foreign].isin(df0[tag_ita])]
    df = df.drop_duplicates()

    # Clean text for both columns
    df[tag_foreign] = _clean_denomination_text(df[tag_foreign])
    df[tag_ita] = _clean_denomination_text(df[tag_ita])

    return df.set_index(tag_foreign)[tag_ita].to_dict()


def get_double_languages_denomination(geo_level, df):
    if geo_level == GeoLevel.COMUNE:
        return _get_double_language_mapping(df, cfg.TAG_COMUNE, cfg.TAG_COMUNE + cfg.TAG_ITA_STRANIERA)
    if geo_level == GeoLevel.PROVINCIA:
        return _get_double_language_mapping(df, cfg.TAG_PROVINCIA, cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA)
    if geo_level == GeoLevel.REGIONE:
        return _get_double_language_mapping(df, cfg.TAG_REGIONE, cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA)


def create_administrative_changes_df():
    path = root_path / Path(cfg.variazioni_amministrative["path"])
    df = pd.read_pickle(path)
    __rename_columns(df, cfg.variazioni_amministrative["column_rename"])
    df = df[df["tipo_variazione"].isin(["ES", "CD"])]
    df["Contenuto del provvedimento"].fillna("", inplace=True)
    df = df[~df["Contenuto del provvedimento"].str.contains("accanto alla denominazione in lingua italiana")]
    # Ensure specific columns have integer types, catching errors
    for tag in [cfg.TAG_CODICE_COMUNE, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_CODICE_REGIONE]:
        if tag in df.columns:
            df[tag] = df[tag].astype(int, errors='raise')
        else:
            raise KeyError(f"Expected column '{tag}' not found in DataFrame.")
    df.to_pickle(root_path / Path(cfg.df_variazioni_mapping["path"]))
    return


def get_administrative_changes_df():
    return pd.read_pickle(root_path / Path(cfg.df_variazioni_mapping["path"]))


def _get_popolazione_df():
    """
    Returns
    Restituisce un dataset contenente la popolazione dei singoli comuni italiani (dato ISTAT)
    """
    path = root_path / Path(cfg.popolazione_comuni["path"])
    df = pd.read_pickle(path / Path(path))
    __rename_columns(df, cfg.popolazione_comuni["column_rename"])
    df = df[df[cfg.TAG_CODICE_COMUNE].notnull()]
    df[cfg.TAG_CODICE_COMUNE] = df[cfg.TAG_CODICE_COMUNE].astype(int)
    return df


def _get_comuni_shape_df():
    """
    Returns
    Restituisce un dataset con le shape di ciascun comune italiano (Provenienza Istat).
    Dataset utilizzato per fare plot geografici dell'italia.
    """
    path = root_path / Path(cfg.shape_comuni["path"])
    last_files, _ = __get_last_shape_file_from_folder(path)
    df = gpd.read_file(path / Path(last_files), encoding='utf-8')
    __rename_columns(df, cfg.shape_comuni["column_rename"])
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
    path = root_path / Path(cfg.shape_province["path"])
    last_files, _ = __get_last_shape_file_from_folder(path)
    df = gpd.read_file(path / Path(last_files), encoding='utf-8')
    __rename_columns(df, cfg.shape_province["column_rename"])
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
    path = root_path / Path(cfg.shape_regioni["path"])
    last_files, _ = __get_last_shape_file_from_folder(path)
    df = gpd.read_file(path / Path(last_files), encoding='utf-8')
    __rename_columns(df, cfg.shape_regioni["column_rename"])
    df["center_x"] = df["geometry"].centroid.x
    df["center_y"] = df["geometry"].centroid.y
    df[cfg.TAG_CODICE_REGIONE] = df[cfg.TAG_CODICE_REGIONE].astype(int)
    return df


def get_df(level):
    if level == GeoLevel.COMUNE:
        return get_df_comuni()

    if level == GeoLevel.PROVINCIA:
        return get_df_province()

    if level == GeoLevel.REGIONE:
        return get_df_regioni()

    else:
        raise Exception("Unknown level")


def _calculate_area_from_shape(df):
    df = gpd.GeoDataFrame(
        df, geometry="geometry"
    )
    df.crs = {'init': "epsg:32632"}
    df[cfg.TAG_SUPERFICIE] = df["geometry"].area
    return pd.DataFrame(df)


def create_df_comuni():
    registry = _get_registry_df()[cfg.registry_comuni["column_rename"].values()]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = registry.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    shape = _get_comuni_shape_df()[cfg.shape_comuni["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_COMUNE)
    df = _calculate_area_from_shape(df)
    df.to_pickle(root_path / Path(cfg.df_comuni["path"]))
    return


def get_df_comuni() -> pd.DataFrame:
    """
        Returns
        A dataframe with all the following details for each comune:
         - Population
         - Area
         - Shape
         - provincia and Regione
    """
    return pd.read_pickle(root_path / Path(cfg.df_comuni["path"]))


def create_df_province():
    registry = _get_registry_df()[
        list(cfg.registry_comuni["column_rename"].values()) + [cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA]]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = registry.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    df = \
    df.groupby([cfg.TAG_PROVINCIA, cfg.TAG_PROVINCIA + cfg.TAG_ITA_STRANIERA, cfg.TAG_CODICE_PROVINCIA, cfg.TAG_SIGLA,
                cfg.TAG_REGIONE, cfg.TAG_AREA_GEOGRAFICA])[
        [cfg.TAG_POPOLAZIONE]].sum().reset_index()
    df = df.replace({'NAN': None})
    shape = _get_province_shape_df()[cfg.shape_province["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_PROVINCIA)
    df = _calculate_area_from_shape(df)
    df.to_pickle(root_path / Path(cfg.df_province["path"]))
    return


def get_df_province() -> pd.DataFrame:
    """
        Returns
        A dataframe with all the following details for each provincia:
         - Population
         - Area
         - Shape
         - Regione
    """
    return pd.read_pickle(root_path / Path(cfg.df_province["path"]))


def create_df_regioni():
    registry = _get_registry_df()[
        list(cfg.registry_comuni["column_rename"].values()) + [cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA]]
    popolazione = _get_popolazione_df()[cfg.popolazione_comuni["column_rename"].values()]
    df = registry.merge(popolazione, how="left", on=cfg.TAG_CODICE_COMUNE)
    df["sigla"].fillna("NAN", inplace=True)
    df = df.groupby(
        [cfg.TAG_REGIONE, cfg.TAG_CODICE_REGIONE, cfg.TAG_REGIONE + cfg.TAG_ITA_STRANIERA, cfg.TAG_AREA_GEOGRAFICA])[
        [cfg.TAG_POPOLAZIONE]].sum().reset_index()
    shape = _get_regioni_shape_df()[cfg.shape_regioni["column_rename"].values()]
    df = df.merge(shape, how="left", on=cfg.TAG_CODICE_REGIONE)
    df = _calculate_area_from_shape(df)
    df.to_pickle(root_path / Path(cfg.df_regioni["path"]))
    return df


def get_df_regioni() -> pd.DataFrame:
    """
        Returns
        A dataframe with all the following details for each Regione:
         - Population
         - Area
         - Shape
    """
    return pd.read_pickle(root_path / Path(cfg.df_regioni["path"]))


def _get_shape_italia():
    df = get_df_regioni()
    df["key"] = "Italia"
    df = gpd.GeoDataFrame(df, geometry="geometry")
    df = df.dissolve(by='key')
    return df


def _download_file(url, destination_path):
    """Download a file from a URL to a specified path."""
    log.info(f"Downloading file from {url}")
    start = datetime.now()
    urllib.request.urlretrieve(url, destination_path)
    log.info(f"Download completed in {datetime.now() - start}")


def _unzip_file(zip_path, extract_to):
    """Unzip a file to the specified directory and delete the original zip file."""
    log.info("Starting file extraction")
    start = datetime.now()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
    log.info(f"Extraction completed in {datetime.now() - start}")


def _process_population_data(file_path, output_path, factor):
    """
    Load, process, and save population data, reducing memory usage.

    Parameters:
    - file_path (Path): Path to the CSV file with population data.
    - output_path (Path): Path to save the processed DataFrame.
    - factor (float): Multiplicative factor for population values.

    Returns:
    pd.DataFrame: Processed population DataFrame.
    """
    df = pd.read_csv(file_path)
    df.columns = ["Lon", "Lat", "Population"]

    # Optimize memory usage by changing data types
    df["Lat"] = df["Lat"].astype('float32')
    df["Lon"] = df["Lon"].astype('float32')
    df["Population"] = (df["Population"] * factor).round(0).astype('int32')

    # Filter out rows with zero population
    df = df[df["Population"] > 0]

    # Save the processed DataFrame as a pickle file
    df.to_pickle(output_path)
    os.remove(file_path)
    return df


def _download_and_prepare_population_density_data():
    """
    Download, extract, process, and save high-resolution population density data for Italy.

    Returns:
    pd.DataFrame: Processed population DataFrame.
    """
    # Paths and configuration setup
    link = cfg.high_resolution_population_density["link"]
    file_name = Path(link).name
    folder_path = root_path / Path(cfg.high_resolution_population_density["folder_path"])
    file_path = folder_path / file_name
    csv_path = folder_path / file_name.replace(".zip", "").replace("_csv", ".csv")
    output_path = root_path / Path(cfg.high_resolution_population_density["file_path"])
    factor = cfg.high_resolution_population_density["multiplicative_factor"]

    # Ensure folder exists
    folder_path.mkdir(parents=True, exist_ok=True)

    # Download and unzip data
    _download_file(link, file_path)
    _unzip_file(file_path, folder_path)

    # Process and save the data
    df = _process_population_data(csv_path, output_path, factor)
    return df


@validate
def get_high_resolution_population_density_df() -> pd.DataFrame:
    file_path = root_path / Path(cfg.high_resolution_population_density["file_path"])
    if os.path.exists(file_path):
        log.info("Start loading the file")
        start = datetime.now()
        df = pd.read_pickle(file_path)
        end = datetime.now()
        log.info("File loaded in {}".format(end - start))
    else:
        df = _download_and_prepare_population_density_data()
    df["Population"] = df["Population"] / cfg.high_resolution_population_density["multiplicative_factor"]
    return df


def remove_high_resolution_population_density_file():
    file_path = root_path / Path(cfg.high_resolution_population_density["file_path"])
    if os.path.isfile(file_path):
        os.remove(file_path)


def create_df():
    create_df_comuni()
    create_df_province()
    create_df_regioni()
    create_administrative_changes_df()


def _fetch_registry_for_year(year: int):
    """
    Downloads and processes the ISTAT registry of Italian comuni for the specified year.

    Parameters:
    - year (int): The target year for the ISTAT registry.

    Returns:
    None: Saves the processed DataFrame as a pickle file.
    """

    link = cfg.registry_comuni["link"]
    path = root_path / Path(cfg.registry_comuni["path"]).with_suffix(".xls")

    # Download the registry file
    _download_file(link, path)

    # Load the Excel file into a DataFrame
    df = pd.read_excel(path, keep_default_na=False)

    # Remove the downloaded Excel file
    os.remove(path.with_suffix(".xls"))

    # Save the DataFrame as a pickle file
    pickle_path = path.with_suffix(".pkl")
    df.to_pickle(pickle_path)

    logging.info(f"Restore previous the registry of the year {year}")
    variation_link = "https://www.anagrafenazionale.interno.it/wp-content/uploads/ANPR_archivio_comuni.csv"
    path, _ = os.path.split(path)
    path = path / Path("ANPR_archivio_comuni.csv")
    _download_file(variation_link, path)
    df_var = pd.read_csv(path)
    df_var["DATAISTITUZIONE"] = pd.to_datetime(df_var["DATAISTITUZIONE"])
    df_var = df_var[
        (df_var["DATAISTITUZIONE"] <= f"{year}-01-01")
        &
        (df_var["DATACESSAZIONE"] > f"{year}-01-01")
    ]
    rename_dict = {
        "DENOMINAZIONE_IT": "Denominazione in italiano",
        "CODISTAT": "Codice Comune formato alfanumerico",
        "IDPROVINCIAISTAT": "Codice Provincia (Storico)(1)",
        "Denominazione (Italiana e straniera)": "Denominazione (Italiana e straniera)",
    }
    df_var["Denominazione (Italiana e straniera)"] = np.where(
        df_var["ALTRADENOMINAZIONE"].notnull(),
        df_var["DENOMINAZIONE_IT"] + "/" + df_var["ALTRADENOMINAZIONE"],
        df_var["DENOMINAZIONE_IT"]
    )

    df_var = df_var[list(rename_dict.keys())]
    df_var.rename(columns=rename_dict, inplace=True)
    df_var["Denominazione in italiano"] = _clean_denomination(df_var["Denominazione in italiano"])
    df_var["Denominazione (Italiana e straniera)"] = _clean_denomination(df_var["Denominazione (Italiana e straniera)"])
    column_merge = "Codice Provincia (Storico)(1)"
    columns_add = list(set(cfg.registry_comuni["column_rename"].keys()) - set(rename_dict.values()))
    columns_add.append(column_merge)
    df = df[columns_add].drop_duplicates()
    df = df_var.merge(df, how="left", on="Codice Provincia (Storico)(1)")

    df.to_pickle(pickle_path)
    return


def _clean_denomination(series):
    """
        Clean and standardize the denomination text in a pandas Series.

        Parameters:
        - series (pd.Series): The Series to be cleaned.

        Returns:
        - pd.Series: The cleaned Series.
        """
    series = series.str.title()

    # Replace specific words with their lowercase equivalents
    words_to_replace = ["di", "della", "sopra", "dei", "da", "sul", "presso", "delle", "del", "degli", "con", "li",
                        "bel", "valle", "val", "nel", "a", "in", "e"]
    for word in words_to_replace:
        series = series.str.replace(f' {word.capitalize()} ', f' {word} ', regex=False)

    # Replace specific prefixes with their lowercase equivalents
    prefixes_to_replace = ["d'", "de'", "dell'", "sull'", "all'"]
    for prefix in prefixes_to_replace:
        series = series.str.replace(f' {prefix.capitalize()}', f'  {prefix}', regex=False)

    return series


def _update_population_info(year):
    path = root_path / Path(cfg.popolazione_comuni["path"])
    istat = sdmx.Request('istat')
    key = dict(FREQ='A', ETA='TOTAL', SESSO="9", STACIVX="99", TIPO_INDDEM="JAN")
    params = dict(startPeriod=f'{year}-01-01', endPeriod=f'{year}-01-01')
    data = istat.data('22_289', key=key, params=params).data
    data = sdmx.to_pandas(data[0]).reset_index()
    data = data[data["ITTER107"].astype(str).str.isnumeric()]
    data = data[["ITTER107", "value"]]
    data.columns = ["Codice Comune", "Popolazione"]
    data.to_pickle(path)


def _update_shape_comuni(year, i=0, force_year=False):
    links = [
        f"{cfg.shape_comuni['link']}0101{year}.zip",
        f"https://www.istat.it/storage/cartografia/confini_amministrativi/non_generalizzati/{year}/Limiti0101{year}.zip"
    ]
    link = links[i]
    file_name = link.split("/")[-1]
    folder_path = (root_path / Path(cfg.shape_comuni["path"])).parent
    file_path = Path(folder_path) / Path(file_name)
    log.info(f"Start downloading the Shape File (63.4M) for the year {year} from link {link}.")
    start = datetime.now()
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(link, file_path)
    except:
        if i < len(links) - 1:
            _update_shape_comuni(year, i=i + 1, force_year=force_year)
            return
        elif not force_year:
            log.info(f"Link for update ISTAT shape file not found (link:{link}). \n"
                     f"ISTAT may hasn't published shape file for {year} yet, try to get file for year {year - 1}.")
            _update_shape_comuni(year - 1, i=0, force_year=True)
            return
        else:
            raise Exception(f"Link for update ISTAT shape file not found (link:{link}). \n"
                            f"Verify internet conenction or ISTAT hasn't published shape file for {year} yet.")
    end = datetime.now()
    log.info(f"Dowload Shape File ended in {end - start}")
    new_file_path = Path(folder_path) / Path(
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
            _folder_path = Path(new_file_path) / Path(_folder)
            shutil.rmtree(root_path / cfg.shape_comuni["path"])
            os.replace(_folder_path, root_path / cfg.shape_comuni["path"])
        elif _folder.startswith("Prov"):
            _folder_path = Path(new_file_path) / Path(_folder)
            shutil.rmtree(root_path / cfg.shape_province["path"])
            os.replace(_folder_path, root_path / cfg.shape_province["path"])
        elif _folder.startswith("Reg"):
            _folder_path = Path(new_file_path) / Path(_folder)
            shutil.rmtree(root_path / cfg.shape_regioni["path"])
            os.replace(_folder_path, root_path / cfg.shape_regioni["path"])

    shutil.rmtree(new_file_path)
    return


def _update_administrative_changes(year):
    path1 = root_path / Path(cfg.variazioni_amministrative["path"].replace("pkl", "zip"))
    link = cfg.variazioni_amministrative["link"]
    urllib.request.urlretrieve(link, path1)
    path2 = root_path / Path(cfg.variazioni_amministrative["path"].replace(".pkl", ""))
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
    df.to_pickle(root_path / Path(cfg.variazioni_amministrative["path"]))
    shutil.rmtree(path2)


def update_data_istat(year=None):
    if year is None:
        year = datetime.now().year
        force_year = False
    else:
        force_year = True
    _fetch_registry_for_year(year)
    _update_shape_comuni(year, force_year=force_year)
    _update_population_info(year)
    _update_administrative_changes(year)
    create_df()
