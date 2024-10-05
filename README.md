# Geo Ita

geo_ita is a Python library for geographical analysis focused on Italy, providing tools to retrieve detailed geographic data and create customized visualizations such as choropleth maps and density estimations.

## Installation

 - Download whl file from [here](https://github.com/stefanogelli92/geo_ita/raw/main/dist/geo_ita-0.0.1-py3-none-any.whl) (63 Mb)
 - Open Terminal on folder where you put the whl file
 - Run the following command:

```bash
pip install geo_ita-0.0.1-py3-none-any.whl
```

## Documentation
1. [Data](#data)
2. [Enrich Dataframe](#enrich-dataframe)
3. [Plot](#plot)

### Data
***
geo_ita provides a variety of datasets sourced from official statistics and research. Below are the key datasets available:
- **Registry** (source [ISTAT](https://www.istat.it/storage/codici-unita-amministrative/Archivio-elenco-comuni-codici-e-denominazioni_Anno_2022.zip) - updated to 01-08-2022): 
       <br>Contains the list of each *Comune* with the hierarchical structure of *Province* and *Regioni*.
- **Surface Area** (source API Istat - updated to 31-12-2021):
       <br>Provides the area (in km²) of each *Comune*.
- **Population** (source API Istat - updated to 31-12-2021): 
       <br>Provides the population of each *Comune*.
- **Shapefiles** (source [ISTAT](https://www.istat.it/storage/cartografia/confini_amministrativi/non_generalizzati/Limiti01012022.zip) - updated to 01-01-2022): 
       <br> Contains shapefiles for each *Comune*, *Provincia* and *Regione*. These are primarily used for plotting [Choropleth map](https://en.wikipedia.org/wiki/Choropleth_map#:~:text=A%20choropleth%20map%20(from%20Greek,each%20area%2C%20such%20as%20population)).
- **High-Resolution Population Density** (source [Facebook](https://data.humdata.org/dataset/italy-high-resolution-population-density-maps-demographic-estimates) - updated to 19-04-2022): 
       <br> Provides a detailed population density estimate derived from Facebook's machine learning models, using satellite imagery to map population distribution at a high resolution.
              
#### Usage
You can retrieve the data from the ISTAT sources with the following three functions: *get_df_comuni*, *get_df_province* and *get_df_regioni* with the 3 different level of aggregation.
```python
from geo_ita.data import get_df_comuni

df_comuni = get_df_comuni()

df_comuni.head(2)

denominazione_comune  codice_comune denominazione_provincia  codice_provincia denominazione_regione  codice_regione sigla  popolazione                                           geometry       center_x      center_y  superficie_km2
               Agliè           1001                  Torino                 1              Piemonte               1    TO         2621  POLYGON ((404703.558 5026682.655, 404733.559 5...  404137.448470  5.024327e+06         13.1464
             Airasca           1002                  Torino                 1              Piemonte               1    TO         3598  POLYGON ((380700.909 4977305.520, 380702.627 4...  380324.100684  4.975382e+06         15.7395 
```
You can also get the list of *Comuni*, *Province* and *Regioni* with *get_comuni_list*, *get_province_list* and *get_regioni_list*.

For High density Population use *get_high_resolution_population_density_df* (the first time it will download it from the source.
***
### Enrich Dataframe
***
The library includes utilities to enrich your DataFrame with geographic and demographic information, adding more value to your analysis.
- **[Geocoding](https://en.wikipedia.org/wiki/Address_geocoding)**: 
        <br>This method returns the geographic coordinates (latitude and longitude) of a location based on its address. Additionally, it enhances the geocoding process by providing information about the corresponding *Comune*, *Provincia*, or *Regione* to assist with accuracy and verification.
        <br>**Note**: The geocoding process may not always successfully determine the coordinates. This is often due to issues such as:
    - Errors or abbreviations in the provided address.
    - Addresses not present in the OpenStreetMap database (used as it is open and free).

    <br>**Performance Warning**: This process requires approximately 1 second per address, which may affect performance when processing large datasets.
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
        <br>This method allows you to enrich a dataset by adding the hierarchical structure of *Province* and *Regioni*, along with the population and area values, based on a given column containing information about the *Comune*, *Provincia*, or *Regione*. It works with various types of data, including the Codice ISTAT, official names (denominazione), or abbreviations (sigla).
        <br>**Note**: When using the name of the *Comune*, it may not always be possible to directly match it with the official ISTAT registry. This method attempts to clean and standardize the input data, handling cases where local names or alternative place names are used instead of the official *Comune* name.
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
# Run first cleaning of columns and try to match it with ISTAT's registry
addinfo.run_simple_match()
# (Optional) The remaining values are searched on OpenStreetMap in order to find any frazione used instead of the name of comune (such as Ostia Lido is a Frazione of the municipality of Rome).
addinfo.run_find_frazioni()
# (Optional) The remaining values are searched on Google in order to find other frazioni used instead of the name of comune.
addinfo.run_find_frazioni_from_google()
# (Optional) The remaining values are searched for similarity with ISTAT's registry. This can find some wrong match so you can look at the match and decide to accept or not this step.
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
- **Geographical DataQuality**:
    <br>This method assesses the quality of a dataset containing geographical information (such as Regione, Provincia, Comune, or coordinates) and returns a list of warnings regarding potential inconsistencies. When possible, the method also suggests corrections to improve the data quality.
```python
# Usage
from geo_ita.enrich_dataframe import GeoDataQuality

df = <you dataframe>

# Create the class and pass the dataframe
dq = GeoDataQuality(df)
# Set alL the columns with geographical information you want to check
dq.set_nazione_tag("nazione")
dq.set_regioni_tag("regione")
dq.set_province_tag("provincia")
dq.set_comuni_tag("comune")
dq.set_latitude_longitude_tag("latitudine", "longitudine")
# Run the check and get the result
result = dq.start_check(show_only_warning=False, sensitive=True)
# Plot an interactive view that can help deep dive into the warning
dq.plot_result()

# Output
```
![plot](./Test/usage_geo_data_quality.PNG?raw=true)

- **Aggregation points by distance**:
        <br>From a given list of point we create groups based on distances.
        <br>There are several approaches to the problem based on what we expect from the groups created. In this method point can be in the same group when their distance is bigger than the entered value, but they have a point in common that has a smaller distance to both of them.
```python
# Usage
from geo_ita.enrich_dataframe import aggregate_point_by_distance

df = pd.DataFrame(data=[[42.000001, 12.000001],
                        [42.000002, 12.000002],
                        [42.001002, 12.001002],
                        [42.001002, 12.001002]], columns=["latitude", "longitude"])

df = aggregate_point_by_distance(df, 1000)

# Output
    latitude       longitude     aggregation_code
  42.0000001      12.0000001                    0
  42.0000002      12.0000002                    0
  42.0010002      12.0010002                    1
  42.0010002      12.0010002                    1
```
- **Population Neraby**:
        <br>From a given list of point and a radius this method add the estimated number of people who live in the nearby. this method use the High Resolution Population Density created by Facebook.
```python
# Usage
from geo_ita.enrich_dataframe import get_population_nearby

df = pd.DataFrame(data=[[42.2463245, 11.2457345],
                        [38.0232362, 12.3242362]], columns=["latitude", "longitude"])
# The first time you use this method you have to download the dataset so this can take additional 2-3 min.
df = get_population_nearby(df, 300)

# Output
    latitude       longitude     n_population
  42.2463245      11.2457345              127  
  38.0232362      12.3242362              402
```


***
### Plot
***
The library supports various types of geographic visualizations. Some plots have two versions: static and interactive (add _interactive on method).
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

## License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)
