# Regional `goSPL` model workflows

For cases where the focus in on regional scale model and not global scale one, we provide a series of notebooks to create input files, run the simulation and finally output some of the results using different workflows. 


## Building inputs for regional model

The first notebooks `1-regionalForcing` and `2-buildInputs` allow to create the input forcing files needed to run a `goSPL` simulation. 
They show how to apply different changes to an initial elevation and impose some tectonic (uplift/subsidence) displacements. 
They also provide a method to impose a paleo-drainage on an initial topography and force the model to fit with a specified paleo-coastline.
All these methods conserve the coordinate reference system (CRS) allowing for simple modifications of the initial topography or conditions and evaluation of `goSPL` outputs.

## Post-processing notebooks

We have defined similar post-processing notebooks as the ones created for the global outputs. 

### Stratigraphic record

Notebook 3 (`3-extractStrata`) extract cross-sections for model ran with stratigraphy turned on.

<img width="936" alt="stratigraphy" src="https://user-images.githubusercontent.com/7201912/144148203-83e5399a-dda4-406c-8115-33af7864f4f0.png">

### Export dataset 

The first notebook `4-exportData` allows to create for each time step series of variables as a `netcdf`. 
These `netcdf` files are then used in the other notebooks.

<img width="936" alt="export" src="https://user-images.githubusercontent.com/7201912/144148609-9470f6fc-4040-41a0-bf62-d4d72f092f92.png">

### Mapping changes

Visualising changes in erosion or deposition in specific regions over simulated time interval can be done with the notebook `5-mappingSed`.

<img width="936" alt="sediment changes" src="https://user-images.githubusercontent.com/7201912/144149015-cb9bd9d9-72ad-4ce2-9403-17c5dec62881.png">

### Extracting geomorphometrics and river drainage basin

Notebooks 6 (`6-extractBasinRivers`) and 7 (`7-plotRivers`) extract some of the geomorphological characteristics of the region as well as river longitudinal profiles for a specific catchment.

<img width="936" alt="basin characteristics" src="https://user-images.githubusercontent.com/7201912/144149441-80bcc128-df24-446a-a8cb-d8c29dfa985f.png">

