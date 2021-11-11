# Global scale paleo-landscape model workflows

The workflows presented here have been designed to build, run and analyse outputs from `goSPL` global models.

https://user-images.githubusercontent.com/7201912/141043940-5d8035be-49a1-4071-ada0-adb972669356.mov

## gospl / Global Scalable Paleo Landscape Evolution  /

**gospl** is an open source, GPL-licensed library providing a scalable parallelised Python-based numerical model to simulate landscapes and basins reconstruction at global scale.

|    |    |
| --- | --- |
| Build Status | [![Linux/MacOS Build Status](https://travis-ci.org/Geodels/gospl.svg?branch=master)](https://travis-ci.org/Geodels/gospl) [![Coverage Status](https://coveralls.io/repos/github/Geodels/gospl/badge.svg?branch=master)](https://coveralls.io/github/Geodels/gospl?branch=master) [![Documentation Status](https://readthedocs.org/projects/gospl/badge/?version=latest)](https://gospl.readthedocs.io/en/latest/?badge=latest)  [![Updates](https://pyup.io/repos/github/Geodels/gospl/shield.svg)](https://pyup.io/repos/github/Geodels/gospl/) |
| Latest release | [![Github release](https://img.shields.io/github/release/Geodels/gospl.svg?label=tag&colorB=11ccbb)](https://github.com/Geodels/gospl/releases) [![PyPI version](https://badge.fury.io/py/gospl.svg?colorB=cc77dd)](https://pypi.org/project/gospl)  |
| Features | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)    [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Geodels/gospl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Geodels/gospl/context:python) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Total alerts](https://img.shields.io/lgtm/alerts/g/Geodels/gospl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Geodels/gospl/alerts/) |

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02804/status.svg)](https://doi.org/10.21105/joss.02804)

## Installation & dependencies (anaconda)

### via Docker

Easiest approach: not ready yet!

### From a terminal

```bash
conda update conda

conda create --name gospl-global python=3.9
conda activate gospl-global
conda install pandas compilers petsc4py llvm-openmp pip netCDF4
conda install mpi4py matplotlib numpy-indexed
conda install scipy scikit-image scikit-learn
conda install h5py pymannkendall seaborn
conda install cartopy geopandas xarray
conda install basemap rioxarray rasterio
conda install meshio ruamel.yaml
conda install cython
conda install pysheds
conda install jupyterlab packaging
conda install pyvista

pip install  vtk stripy
pip install pyyaml==5.1
pip install  pyvista cmocean
pip install  richdem descartes
pip install  pyevtk itkwidgets

conda install scikit-fuzzy pre-commit
conda install imageio-ffmpeg imageio
conda install qt

conda install gmt==6.2.0

conda install pygmt
conda install pygplates

conda install numba mesplex
pip install gospl
```


### Using an environment.yml file

Copy and paste the following `environment.yml` file somewhere on your local hard drive:

    name: gospl-global
    channels:
      - conda-forge
      - defaults
      - anaconda
    dependencies:
      - python=3.9
      - compilers 
      - numpy 
      - pandas 
      - petsc4py 
      - llvm-openmp 
      - pip 
      - netCDF4
      - mpi4py 
      - matplotlib 
      - numpy-indexed
      - scipy 
      - scikit-image 
      - scikit-fuzzy
      - scikit-learn
      - h5py 
      - pymannkendall 
      - seaborn
      - cartopy 
      - geopandas
      - xarray
      - basemap 
      - rioxarray 
      - rasterio
      - meshio 
      - ruamel.yaml 
      - cython 
      - pysheds 
      - jupyterlab 
      - packaging
      - pyvista
      - pre-commit
      - imageio-ffmpeg 
      - imageio
      - qt
      - gmt==6.2.0
      - pygmt
      - pygplates
      - numba 

      - pip:
        - pyyaml==5.1
        - vtk
        - stripy
        - meshplex            
        - gospl


(**Note:** Installing this environment will also install many dependencies, including compiled libraries. This is totally fine; even if you have these libraries already installed through your system package manager, `conda` will install and link for use in the environment a configuration which should be to play nicely and work with all of its components.)

Create this environment through `conda`

    $ conda update conda
    $ conda env create -f environment.yml
    
Activate this environment

    $ source activate gospl-global
    
There is also an additional library that will need to be installed called `InitialisingEarth` available from:
+ [here](https://github.com/suoarski/InitialisingEarth.git) in the branch named `package` 
+ to install the library you will need to look at the `README` file in the github repository

*This environment should be sufficient for all of the presented examples in this documentation.*

## Workflows

Example of 100 Ma simulation of landscape dynamic accounting for erosion and deposition using `goSPL`.

https://user-images.githubusercontent.com/7201912/141035954-b0ccc9a7-7b25-4a9f-affd-53dc371b0870.mov

### Pre-processing

The first 2 folders: `1-data` & `2-inputgen` are used to perform some of the pre-processing steps required to create the input files for `goSPL`.

Folder `1-data` contains the high-resolution paleo-elevation from Scotese and the associated rainfall map from Valdes et al. as well as the velocities from the plate reconstruction models. All the dataset are available either from the litterature or extracted from the Gplates software. As an example a series of initial forcing input files based on Scotese paleomaps and paleo-climate for the last 100 Ma are provided as a `doi` from figshare.

Folder `2-inputgen` takes the data files from folder `1-data` to build the `goSPL` inputs. For a specified resolution it creates the initial unstructured elevation mesh (`mesh_X_XXMa.npz`), the rainfall for considered period of time (`rain_X_XXMa.npz`), the `goSPL` nodes horizontal displacements based on considered plate tectonic model (`plate_X_XXMa.npz`), as well as the geometric tectonic forcing conditions (`tecto_X_XXMa.npz`). To build these input files, we rely on a library [InitialisingEarth](https://github.com/suoarski/InitialisingEarth.git) that needs to be installed separetly (see previous section).

### Constrained/unconstrained models

The simulation folder `3-simulations` provides some examples on how to run `goSPL` model at global scale with different approaches to constrain the simulation with paleo-elevation model. Three `goSPL` input files are given and we give 2 notebooks that could be used to better *constrained*, *unconstrained* and *missmatch* approaches. 

### Post-processing

The `4-analysis` folder contains a series of notebooks to analyse some of `goSPL` outputs. It can be used to:

1. Evaluate major drainage systems
2. Estimate geomorphometrics through space and time
3. Plot longitudinal rivers profiles
4. Analyse stratigraphic record

