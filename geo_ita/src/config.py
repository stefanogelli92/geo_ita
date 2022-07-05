USER_AGENT = "geo_ita"

TAG_COMUNE = "denominazione_comune"
TAG_CODICE_COMUNE = "codice_comune"
TAG_PROVINCIA = "denominazione_provincia"
TAG_CODICE_PROVINCIA = "codice_provincia"
TAG_SIGLA = "sigla"
TAG_REGIONE = "denominazione_regione"
TAG_CODICE_REGIONE = "codice_regione"
TAG_POPOLAZIONE = "popolazione"
TAG_SUPERFICIE = "superficie_km2"
TAG_AREA_GEOGRAFICA = "area_geografica"
TAG_ITA_STRANIERA = "_ita_straniera"
LEVEL_COMUNE = 0
LEVEL_PROVINCIA = 1
LEVEL_REGIONE = 2
CODE_CODICE_ISTAT = 0
CODE_SIGLA = 1
CODE_DENOMINAZIONE = 2
KEY_UNIQUE = "key_geo_ita"
UNIQUE_TAG = "_geo_ita"

anagrafica_comuni = {
    "path": r"data_sources/Anagrafica",
    "file_name": "anagrafica.pkl",
    "column_rename": {"Denominazione in italiano": TAG_COMUNE,
                      "Denominazione (Italiana e straniera)": TAG_COMUNE + TAG_ITA_STRANIERA,
                      "Codice Comune formato alfanumerico": TAG_CODICE_COMUNE,
                      """Denominazione dell'Unità territoriale sovracomunale 
(valida a fini statistici)""": TAG_PROVINCIA,
                       "Codice Provincia (Storico)(1)": TAG_CODICE_PROVINCIA,
                       "Denominazione Regione": TAG_REGIONE,
                       "Codice Regione": TAG_CODICE_REGIONE,
                       "Sigla automobilistica": TAG_SIGLA,
                       "Ripartizione geografica": TAG_AREA_GEOGRAFICA}}

variazioni_amministrative = {
    "path": r"data_sources/Variazioni",
    "column_rename": {"Denominazione Comune": TAG_COMUNE,
                      "Denominazione Comune associata alla variazione o nuova denominazione": "new_denominazione_comune",
                      "Data decorrenza validità amministrativa": "data_decorrenza",
                       "Tipo variazione": "tipo_variazione"}}

popolazione_comuni = {
    #"path": r"data_sources/Comuni/Popolazione",
    "path": r"data_sources/Comuni/Popolazione/popolazione.pkl",
    "column_rename": {"Codice Comune": TAG_CODICE_COMUNE,
                      "Popolazione": TAG_POPOLAZIONE}
}

shape_comuni = {
    "link": r"https://www.istat.it/storage/cartografia/confini_amministrativi/non_generalizzati/Limiti",
    "path": r"data_sources/Comuni/Shape",
    "column_rename": {"PRO_COM": TAG_CODICE_COMUNE,
                      "geometry": "geometry",
                      "center_x": "center_x",
                      "center_y": "center_y"}
}

shape_province = {
    "path": r"data_sources/Province/Shape",
    "column_rename": {"COD_PROV": TAG_CODICE_PROVINCIA,
                      "geometry": "geometry",
                      "center_x": "center_x",
                      "center_y": "center_y"}
}

shape_regioni = {
    "path": r"data_sources/Regioni/Shape",
    "column_rename": {"COD_REG": TAG_CODICE_REGIONE,
                      "geometry": "geometry",
                      "center_x": "center_x",
                      "center_y": "center_y"}
}

dimensioni_comuni = {
    "path": r"data_sources/Comuni/Dimensioni/dimensioni.pkl",
    "column_rename": {"Codice Comune": TAG_CODICE_COMUNE,
                      "Superficie totale (Km2)": TAG_SUPERFICIE}
}

df_comuni = {
    "path": r"data_sources/df_comuni.pkl"}

df_province = {
    "path": r"data_sources/df_province.pkl"}

df_regioni = {
    "path": r"data_sources/df_regioni.pkl"}

df_variazioni_mapping = {
    "path": r"data_sources/df_variazioni_mapping.pkl"}

min_acceptable_similarity = 0.85

clear_den_replace = [(" di ", " "),
                     (" nell", " "),
                     (" nel", " "),
                     (" in ", " "),
                     ("l ", "l"),
                     ("d ", "d"),
                     ("sant ", "sant"),
                     ("s ", "san "),
                     ("santa", "san"),
                     ("santo", "san"),
                     ("sant", "san")]

clear_provnce_name = {
    "citta metropolitana di ": "",
    "provincia di ": "",
    "provincia del ": "",
    "province of ": "",
    "provincia autonoma di ": "",
    "libero consorzio comunale di ": "",
    "metropolitan city of ": ""
}

clear_comuni_name = {
    "comune di ": "",
    "localita di ": "",
    "localita ": "",
    "frazione di ": "",
    "frazione ": ""
}

rename_english_name = {"rome": "roma",
                       "milan": "milano",
                       "naples": "napoli",
                       "turin": "torino",
                       "florence": "firenze",
                       "venice": "venezia",
                       "padua": "padova",
                       "syracuse": "siracusa",
                       "bozen": "bolzano",
                       "south sardinia": "sud sardegna",
                       "south tyrol": "bolzano",
                       "sicily": "sicilia",
                       "piedmont": "piemonte",
                       "aosta valley": "valle d aosta",
                       "tuscany": "toscana",
                       "apulia": "puglia",
                       "lombardy": "lombardia",
                       "sardinia": "sardegna"
                       }

comuni_exceptions = {
    "paterno paterno": "paternò",
    "paternò": "paterno paterno"
}


comuni_omonimi = {
    TAG_COMUNE: ["Livo", "Livo",
                 "Peglio", "Peglio",
                 "Castro", "Castro",
                 "Samone", "Samone",
                 "Calliano", "Calliano",
                 "San Teodoro", "San Teodoro"],
    TAG_SIGLA: ["CO", "TN",
                "CO", "PU",
                "BG", "LE",
                "TO", "TN",
                "AT", "TN",
                "ME", "SS"],
    TAG_PROVINCIA: ["Como", "Trento",
                    "Como", "Pesaro e Urbino",
                    "Bergamo", "Lecce",
                    "Torino", "Trento",
                    "Asti", "Trento",
                    "Messina", "Sassari"],
    TAG_REGIONE: ["Lombardia", "Trentino-Alto Adige",
                  "Lombardia", "Marche",
                  "Lombardia", "Puglia",
                  "Piemonte", "Trentino-Alto Adige",
                  "Piemonte", "Trentino-Alto Adige",
                  "Sicilia", "Sardegna"],
    "new_name": ["Livo como", "Livo",
                 "Peglio como", "Peglio",
                 "Castro bergamo", "Castro",
                 "Samone", "Samone trento",
                 "Calliano asti", "Calliano",
                 "San Teodoro messina", "San Teodoro"]
}

high_resolution_population_density = {
    "folder_path": "data_sources/HighResolutionPopulationDensity",
    "file_path": "data_sources/HighResolutionPopulationDensity/high_resolution_population_density.pkl",
    #"link": r"https://data.humdata.org/dataset/0eb77b21-06be-42c8-9245-2edaff79952f/resource/1e96f272-7d86-4108-b4ca-5a951a8b11a0/download/population_ita_2019-07-01.csv.zip",
    "link": r"https://data.humdata.org/dataset/0eb77b21-06be-42c8-9245-2edaff79952f/resource/1e96f272-7d86-4108-b4ca-5a951a8b11a0/download/ita_general_2020_csv.zip",
    "moltiplicative_factor": 10,
}

regex_find_frazioni = r"({}) ?(\[[^\[.]+\])? ?(\([^\(.]+\))? ?([^.]{{0,40}}) (si trova nel comune|fa parte del comune|frazione ?(\([^.]+\))? ?[^.]{{0,20}} ?(\([^.]+\))? comune|borgo del comune|località del comune) (di ?)([^.]+)"
list_road_prefix = ["via", "viale", "strada", "st", "largo", "corso", "contrada"]

simplify_values={LEVEL_REGIONE: 500,
                 LEVEL_PROVINCIA: 500,
                 LEVEL_COMUNE: 250}

google_search_api_key = "AIzaSyDQTlp0F0tExsXg1PZn1FF_ugK_bwBhlqA"
google_search_cse_id = "8558e6507a2bf9d77"

