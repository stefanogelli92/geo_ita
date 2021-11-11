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
LEVEL_COMUNE = 0
LEVEL_PROVINCIA = 1
LEVEL_REGIONE = 2
CODE_CODICE_ISTAT = 0
CODE_SIGLA = 1
CODE_DENOMINAZIONE = 2
KEY_UNIQUE = "key_geo_ita"
KEY_UNIQUE2 = "key_geo_ita2"

anagrafica_comuni = {
    "path": r"data_sources/Anagrafica",
    "file_name": "anagrafica.pkl",
    "column_rename": {"Denominazione in italiano": TAG_COMUNE,
                      "Denominazione (Italiana e straniera)": "denominazione_comune_ita_straniera",
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
    "path": r"data_sources/Comuni/Popolazione",
    "column_rename": {"ITTER107": TAG_CODICE_COMUNE,
                      "Value": TAG_POPOLAZIONE}
}

shape_comuni = {
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
    "path": r"data_sources/Comuni/Dimensioni",
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

rename_comuni_nomi = {"rome": "roma",
                      "milan": "milano",
                      "naples": "napoli",
                      "turin": "torino",
                      "florence": "firenze",
                      "venice": "venezia",
                      "padua": "padova",
                      "syracuse": "siracusa",
                      "bozen": "bolzano"
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
    "link": r"https://data.humdata.org/dataset/0eb77b21-06be-42c8-9245-2edaff79952f/resource/1e96f272-7d86-4108-b4ca-5a951a8b11a0/download/population_ita_2019-07-01.csv.zip"
}

regex_find_frazioni = r"({}) ?(\[[^\[.]+\])? ?(\([^\(.]+\))? ?([^.]{{0,40}}) (si trova nel comune|fa parte del comune|frazione ?(\([^.]+\))? ?[^.]{{0,20}} ?(\([^.]+\))? comune|borgo del comune|località del comune) (di ?)([^.]+)"
list_road_prefix = ["via", "viale", "strada", "st", "largo", "corso", "contrada"]