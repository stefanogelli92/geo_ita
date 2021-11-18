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
2. [Enrich Dataframe](#enrich-dataframe)
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
You can import all the informations listed before with 3 functions: *get_df_comuni*, *get_df_province* and *get_df_regioni* with the 3 different level of aggregation.
```python
from geo_ita.data import get_df_comuni

df_comuni = get_df_comuni()

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
        <br>Warning: This process takes about 1 second per element.
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
        <br>There are two version of reverse geocoding:
        <br> 1. **get_address_from_coordinates**: This method will return the address of a place from the coordinates. Warning: This process takes about 1 second per element.
        <br> 2. **get_city_from_coordinates**: This method will return the comune, provincia ane regione from the coordinates.
```python
# Usage
from geo_ita.enrich_dataframe import get_address_from_coordinates, get_city_from_coordinates

df = pd.DataFrame(data=[[41.939965,  12.470965],
                        [45.460908,   9.191498],
                        [40.846269,  14.267525]], columns=["latitude", "longitude"])

df = get_address_from_coordinates(df, latitude_columns="latitude", longitude_columns="longitude")

# Output

   latitude  longitude                     address      city
  41.939965  12.470965    'corso di francia, roma'    'Roma'
  45.460908   9.191498         'via larga, Milano'  'Milano'
  40.846269  14.267525  'via nuova marina, Napoli'  'Napoli'

df = get_city_from_coordinates(df, latitude_columns="latitide", longitude_columns="longitude")

# Output

   latitude  longitude    denominazione_comune    denominazione_provincia    sigla   denominazione_regione
  41.939965  12.470965                   'Roma'                     'Roma'     'RM'                 'Lazio'
  45.460908   9.191498                 'Milano'                   'Milano'     'MI'             'Lombardia'
  40.846269  14.267525                 'Napoli'                   'Napoli'     'NA'              'Campania'
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
# (Optional) The remaining values are searched on Google in order to find other frazioni used instead of the name of comune.
addinfo.run_find_frazioni_from_google()
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
from geo_ita.data import get_df_regioni, get_df_province
from geo_ita.plot import plot_choropleth_map_regionale, plot_choropleth_map_provinciale_interactive

# Get the dataframe you want to use
df = get_df_regioni()
   denominazione_regione  superficie_km2  popolazione
                 Abruzzo      10831.8388      1293941
              Basilicata      10073.3226       553254
                Calabria      14706.3858      1894110

# Simple use plot
plot_choropleth_map_regionale(df, region_tag='denominazione_regione', value_tag='popolazione')
```
![plot](./Test/usage_choropleth_regionale.png?raw=true)
```python
# Interactive plot of a single Regione
df = get_df_province()

plot_choropleth_map_provinciale_interactive(df, 
                                            'denominazione_regione', 
                                            {"popolazione": "Popolazione",
                                             "superficie_km2": "Superficie"},
                                            filter_regione="Toscana",
                                            title="Toscana")
```
![plot](./Test/usage_choropleth_provinciale_interactive.png?raw=true)
- **Point Map**:
        <br>This method will plot a list of point in a map. The are also the interactive version were you can also read different information in each point.

```python
# Usage
from geo_ita.data import get_df_comuni
from geo_ita.plot import plot_point_map, plot_point_map_interactive

# Get the dataframe you want to use
df = get_df_province()[["denominazione_comune", "center_x", "center_y", "popolazione"]]

    denominazione_comune       center_x      center_y  popolazione
                   Agliè  404137.448470  5.024327e+06         2621
                 Airasca  380324.100684  4.975382e+06         3598
            Ala di Stura  365344.513419  5.018472e+06          441

# Simple use plot
plot_point_map(df, latitude_columns='center_y', longitude_columns='center_x', title="Province")
```
![plot](./Test/usage_point_map_comuni.png?raw=true)

- **[Density Estimation](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation)**:
        <br>This method will plot a density on map.
        <br>There are two type of density estimation:
        <br> 1. Simple point density: This will show where point are more or less concentrated. In this case you need to pass only the coordinates of points.
        <br> 2. Density of a variable in each point: This will show the density of the variable. In this case you need to pass the coordinates and the value of the variable of each point.

```python
# Usage
from geo_ita.plot import plot_kernel_density_estimation

# Get the dataframe you want to use 
df = get_df_comuni()

# Simple point density
plot_kernel_density_estimation(df, latitude_columns='center_y', longitude_columns='center_x',
                               n_grid_x=500, n_grid_y=500)
```
![plot](./Test/usage_kernel_density_simple.png?raw=true)
```python
# Density of variable Popolazione
plot_kernel_density_estimation_interactive(df, value_tag="popolazione",
                               latitude_columns='center_y', longitude_columns='center_x',
                               n_grid_x=500, n_grid_y=500)
```
![plot](./Test/usage_kernel_density_variable.png?raw=true)
***
## TODO
1. Condividere libreria senza whl
2. Completare test Units
3. Aggiornamento dati ISTAT
4. Aggioranmento dati automatico
5. Valutare utilizzo dei dati storici ISTAT e non solo ultimi
6. Trovare nome più parlante per Enrich Dataset
7. Utilizzo possibile delle api di Google con codice in input
8. Semplificare utilizzo AddGeoInfo (forse non classe)
9. Consigliare step successivo (AddGeoInfo)
10. Point Map inserire possibilità di plottare più dataset di punti (e forse anche non solo punti)
11. Kernel Estimation Migliorare guida
12. Kernel estimation semplificare utilizzo
13. Possiblità di aggiungere densità demografica da dataset Facebook in un modulo (o forse da guida ma troppo complesso)
14. Creare calcolo puntuale Kernel estimation per informazioni puntuali e non plot
15. valutare calcolo distanza da costa
16. Valutare distanza in linea d'aria da uscite autostrada + plot autostrade + uscite
17. Da valutare calcolo distanza tra punti su strada (è corretto averlo in questa libreria?)

## License
[MIT](https://choosealicense.com/licenses/mit/)
