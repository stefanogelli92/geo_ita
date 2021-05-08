# Geo Ita

geo_ita is a library for geographical analysis within Italy.

## Installation

WARNING: Da rivedere!

 - Download whl file from [here]() (66 Mb)
 - Open Terminal on folder where you put the whl file

```bash
pip install geo_ita-0.0.38-py3-none-any.whl
```

## Documentation
1. [Data](#data)
2. [Enrich Dataset](#enrich_dataset)
3. [Plot](#plot)

### Data
***
Here you can get all the data avaiable in this libraries.
- **Anagrafica** (source [ISTAT](https://www.istat.it/it/archivio/6789) - updated to 31-12-2020): 
       <br>Contains the list of each *Comune* with the hierarchical structure of *Province* and *Regioni*.
- **Superficie** (source [ISTAT](https://www.istat.it/it/files//2015/04/Superfici-delle-unit%C3%A0-amministrative-Dati-comunali-e-provinciali.zip) - updated to 09-10-2011):
       <br>Contains the area of each *Comune* in Km2.
- **Popolazione** (source [ISTAT](http://dati.istat.it/Index.aspx?DataSetCode=DCIS_POPRES1) - updated to 01-01-2020): 
       <br>Contains the population of each *Comune*.
- **Shape** (source [ISTAT](https://www.istat.it/it/archivio/222527) - updated to 01-01-2020): 
       <br> Contains the shapes of each *Comune*, *Provincia* and *Regione*. The main use of the shapes is to plot the [Choropleth map](https://en.wikipedia.org/wiki/Choropleth_map#:~:text=A%20choropleth%20map%20(from%20Greek,each%20area%2C%20such%20as%20population)).
#### Usage
You can import all the informations listed before with 3 functions: *create_df_comuni*, *create_df_province* and *create_df_regioni* with the 3 different level of aggregation.
```python
from geo_ita.src._data import create_df_comuni

df_comuni = create_df_comuni()

df_comuni.head(2)

denominazione_comune  codice_comune denominazione_provincia  codice_provincia denominazione_regione  codice_regione sigla  popolazione                                           geometry       center_x      center_y  superficie_km2
               Agliè           1001                  Torino                 1              Piemonte               1    TO         2621  POLYGON ((404703.558 5026682.655, 404733.559 5...  404137.448470  5.024327e+06         13.1464
             Airasca           1002                  Torino                 1              Piemonte               1    TO         3598  POLYGON ((380700.909 4977305.520, 380702.627 4...  380324.100684  4.975382e+06         15.7395 
```
***
### Enrich Dataset
***
Here you can find some methods that add some geografical information to your dataset.
- **get_province_from_city**: 
        <br>From a given columns with the information of *Comune* (works both with Codice ISTAT or denominazione), add the hierarchical structure of *Province* and *Regioni*.
        <br>Warning: If you use the name of *Comune* it is not always possible to match the name present in the anagrafica ISTAT (See FAQ).
- **get_city_from_coordinates**:    
        From each pair of coordinates (the library try to atomatically detect the coord system used) add the informations of *Comune*, *Provincia* and *Regione*.
- **get_coordinates_from_address**: 
        <br>This will return the coordinates of a place from the address. You can add information of *Comune*, *Provincia* or *Regione* in order to help the search and test the result.
        <br>Warning: This process cannot always determine the coordinates. Read the FAQ to better understand limitation and tips for a better result.
***
### Plot
***

***
```python
from geo_ita.src._enrich_dataframe import get_coordinates_from_address

df = get_coordinates_from_address(df, "address")
```

## License
[MIT](https://choosealicense.com/licenses/mit/)