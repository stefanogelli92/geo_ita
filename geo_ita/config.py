

anagrafica_comuni = {
    "path": r"data_sources/Anagrafica",
    "column_rename": {"Denominazione in italiano": "denominazione_comune",
                      "Codice Comune formato alfanumerico": "codice_comune",
                      """Denominazione dell'Unità territoriale sovracomunale 
(valida a fini statistici)""": "denominazione_provincia",
                       "Codice Provincia (Storico)(1)": "codice_provincia",
                       "Denominazione Regione": "Regione",
                       "Codice Regione": "codice_regione",
                       "Sigla automobilistica": "sigla"}}

popolazione_comuni = {
    "path": r"data_sources/Comuni/Popolazione",
    "column_rename": {"Territorio": "denominazione_comune",
                      "ITTER107": "codice_comune",
                      "Value": "popolazione"}
}

shape_comuni = {
    "path": r"data_sources/Comuni/Shape",
    "column_rename": {"PRO_COM": "codice_comune",
                      "geometry": "geometry"}
}

shape_province = {
    "path": r"data_sources/Province/Shape",
    "column_rename": {"COD_PROV": "codice_provincia",
                      "geometry": "geometry"}
}

shape_regioni = {
    "path": r"data_sources/Regioni/Shape",
    "column_rename": {"COD_REG": "codice_regione",
                      "geometry": "geometry"}
}

dimensioni_comuni = {
    "path": r"data_sources/Comuni/Dimensioni",
    "column_rename": {"Codice Regione": "codice_regione",
                      "Codice Provincia": "codice_provincia",
                      "Codice Comune": "codice_comune",
                      "Superficie totale (Km2)": "superficie_km2"}
}