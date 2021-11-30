## Global scale landscape evolution post-processing

We have created a series of workflows to extract some of the information from `goSPL` outputs. 

Below is a list of some of these numerical recipes and outputs that could be automatically generated.

The first notebook `1-exportData` allows to create for each time step series of variables as a `netcdf`. These `netcdf` files are then used in the other notebooks. 

### Evaluation of major drainage systems

Notebook 2 (`2-riverFluxes`) extract river water and sediment fluxes over time.

<img width="754" alt="sedflux" src="https://user-images.githubusercontent.com/7201912/141053541-b3d92299-fa61-4a9a-b8d4-a3254dfb5d03.png">

### Estimate geomorphometrics through space and time

Notebook 3 (`3-regionalPlot`) & Notebook 4 (`4-basinMorpho`) allow to focus on specific regional extent and compute some of the geomorphometrics that characterise the landscape.

<img width="901" alt="region" src="https://user-images.githubusercontent.com/7201912/141054661-54966a82-9140-461f-b2b3-bcd31a8a6110.png">

### Longitudinal rivers profiles

Notebook 5 (`5-plotRiver`) extract longitudinal profiles characteristics for a specific catchment.

<img width="936" alt="rivers" src="https://user-images.githubusercontent.com/7201912/141054675-a71fc160-3cf7-491c-ac74-e0c24876c291.png">

### Stratigraphic record

Notebook 6 (`6-extractStrata`) extract cross-sections for model ran with stratigraphy turned on.

<img width="951" alt="strati" src="https://user-images.githubusercontent.com/7201912/141054646-5d992ef8-3c06-4998-bb38-dc2d221ae158.png">
