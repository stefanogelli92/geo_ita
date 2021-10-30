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
2. [Enrich Dataframe](#Enrich Dataframe)
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
from geo_ita.data import create_df_comuni

df_comuni = create_df_comuni()

df_comuni.head(2)

denominazione_comune  codice_comune denominazione_provincia  codice_provincia denominazione_regione  codice_regione sigla  popolazione                                           geometry       center_x      center_y  superficie_km2
               Agliè           1001                  Torino                 1              Piemonte               1    TO         2621  POLYGON ((404703.558 5026682.655, 404733.559 5...  404137.448470  5.024327e+06         13.1464
             Airasca           1002                  Torino                 1              Piemonte               1    TO         3598  POLYGON ((380700.909 4977305.520, 380702.627 4...  380324.100684  4.975382e+06         15.7395 
```
You can also get the list of *Comuni*, *Province* and *Regioni* with *get_list_comuni*, *get_list_province* and *get_list_regioni*.
***
### Enrich Dataframe
***
Here you can find some methods that add some geografical information to your dataset.
- **[Geocoding](https://en.wikipedia.org/wiki/Address_geocoding)**: 
        <br>This method will return the coordinates of a place from the address. It will also add informations about *Comune*, *Provincia* or *Regione* in order to help the search and test the result.
        <br>Warning: This process cannot always determinate the coordinates. This is mostly due to an address with some errors or abbreviation, or some address that are not in the OpenStreetMaps project (we use it because it is open and free).
```python
# Usage
from geo_ita.enrich_dataframe import get_coordinates_from_address

df = pd.DataFrame(data=[["Corso di Francia, Roma"],
                        ["Via Larga, Milano"],
                        ["Via NUova Marina, Napoli"]], columns=["address"])

df = get_coordinates_from_address(df, "address")

# Output

                    address   latitude  longitude
     Corso di Francia, Roma  41.939965  12.470965
          Via Larga, Milano  45.460908   9.191498
   Via NUova Marina, Napoli  40.846269  14.267525 
```
- **[Reverse Geocoding](https://en.wikipedia.org/wiki/Reverse_geocoding)**: 
        <br>This method will return the address of a place from the coordinates.
```python
# Usage
from geo_ita.enrich_dataframe import get_address_from_coordinates

df = pd.DataFrame(data=[[41.939965,  12.470965],
                        [45.460908,   9.191498],
                        [40.846269,  14.267525]], columns=["latitide", "longitude"])

df = get_address_from_coordinates(df, latitude_columns="latitide", longitude_columns="longitude")

# Output

   latitude  longitude                     address      city
  41.939965  12.470965    'corso di francia, roma'    'Roma'
  45.460908   9.191498         'via larga, Milano'  'Milano'
  40.846269  14.267525  'via nuova marina, Napoli'  'Napoli'
```
- **AddGeographicalInfo**:
        <br>From a given columns with the information of *Comune*, *Provincia* or *Regione* (works both with Codice ISTAT, denominazione or sigla), add the hierarchical structure of *Province* and *Regioni* and the value of population and area.
        <br>Warning: If you use the name of *Comune* it is not always possible to match the name present in the anagrafica ISTAT, this method try to clean the value and find possible località used instead of the name of *comune*.
```python
# Usage
from geo_ita.enrich_dataframe import AddGeographicalInfo

df = pd.DataFrame(data=[["Milano", "Milano", "Milano", "MI", "Lombardia"],
                        ["Florence", "Firenze", "Firenze", "FI", "Toscana"],
                        ["florence", "Firenze", "Firenze", "FI", "Toscana"],
                        ["porretta terme", "Alto Reno Terme", "Bologna", "BO", "Emilia romagna"]], columns=["Citta", "comune", "provincia", "sl", "regione"])
# Create the class and pass the dataframe
addinfo = AddGeographicalInfo(df)
# Set al least one column with the info of comune, provincia or regione (the column can contains the name or the ISTAT code or sigla, the method will automatically detect and use  it)
addinfo.set_comuni_tag("Citta")
# Run first cleaning of columns and try to match it with ISTAT's anagrafica
addinfo.run_simple_match()
# (Optional) The remaining values are searched on OpenStreetMap in order to find any frazione used instead of the name of comune (such as Ostia Lido is a Frazione of the municipality of Rome).
addinfo.run_find_frazioni()
# (Optional) The remaining values are searched for similarity with ISTAT's anagrafica. This can find some wrong match so you can look at the match and decide to accept or not this step.
addinfo.run_similarity_match()
# (Optional) You can show the similarity step result in order to accept or decline the step
print(addinfo.get_similairty_result())
# (Optional) Accept the similarity step
addinfo.accept_similarity_result()
# (Optional) You can also use a manual match if you find manually any
addinfo.use_manual_match({"Milanoa": "Milano"})
# Get Result
result = addinfo.get_result()

# Output

           Citta           comune provincia  sl         regione  popolazione    codice_comune  codice_provincia denominazione_comune  superficie_km2  denominazione_provincia  codice_regione area_geografica sigla  denominazione_regione
          Milano           Milano    Milano  MI       Lombardia      1406242            15146                15               Milano        181.6727                   Milano               3      Nord-ovest    MI              Lombardia
        Florence          Firenze   Firenze  FI         Toscana       366927            48017                48              Firenze        102.3187                  Firenze               9          Centro    FI                Toscana
        florence          Firenze   Firenze  FI         Toscana       366927            48017                48              Firenze        102.3187                  Firenze               9          Centro    FI                Toscana
  porretta terme  Alto Reno Terme   Bologna  BO  Emilia romagna         6953            37062                37      Alto Reno Terme             NaN                  Bologna               8        Nord-est    BO         Emilia-Romagna
```

***
### Plot
***
Here you can find some methods that show some useful plot from a dataframe with geographical information. Some plots have two versions: static and interactive (add _interactive on method).
- **[Choropleth Map](https://en.wikipedia.org/wiki/Choropleth_map)**:
        <br>This method show a plot where regioni, province or comuni are colored based on a specific value (numerical or chategorial).
        <br> You can use one of them:
            <br> 1. Regione: plot_choropleth_map_regionale - plot_choropleth_map_regionale_interactive
            <br> 2. Provincia: plot_choropleth_map_provinciale - plot_choropleth_map_provinciale_interactive
            <br> 3. Comune: plot_choropleth_map_comunale - plot_choropleth_map_comunale_interactive


```python
# Usage
from geo_ita.data import create_df_regioni
from geo_ita.plot import plot_choropleth_map_regionale, plot_choropleth_map_comunale_interactive

# Get the dataframe you want to use
df = create_df_regioni()



```

## License
[MIT](https://choosealicense.com/licenses/mit/)