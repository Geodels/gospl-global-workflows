# Global scale paleo-landscape model workflows

The workflows presented here have been designed to build, run and analyse outputs from `goSPL` global models.

<img width="960" alt="pres" src="https://user-images.githubusercontent.com/7201912/141036579-0cf367e0-9ce8-47a8-8e1d-b5d5b1b0e1d6.png">

## gospl / Global Scalable Paleo Landscape Evolution  /

**gospl** is an open source, GPL-licensed library providing a scalable parallelised Python-based numerical model to simulate landscapes and basins reconstruction at global scale.

|    |    |
| --- | --- |
| Build Status | [![Linux/MacOS Build Status](https://travis-ci.org/Geodels/gospl.svg?branch=master)](https://travis-ci.org/Geodels/gospl) [![Coverage Status](https://coveralls.io/repos/github/Geodels/gospl/badge.svg?branch=master)](https://coveralls.io/github/Geodels/gospl?branch=master) [![Documentation Status](https://readthedocs.org/projects/gospl/badge/?version=latest)](https://gospl.readthedocs.io/en/latest/?badge=latest)  [![Updates](https://pyup.io/repos/github/Geodels/gospl/shield.svg)](https://pyup.io/repos/github/Geodels/gospl/) |
| Latest release | [![Github release](https://img.shields.io/github/release/Geodels/gospl.svg?label=tag&colorB=11ccbb)](https://github.com/Geodels/gospl/releases) [![PyPI version](https://badge.fury.io/py/gospl.svg?colorB=cc77dd)](https://pypi.org/project/gospl) [![DOI](https://zenodo.org/badge/206898115.svg)](https://zenodo.org/badge/latestdoi/206898115) |
| Features | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)    [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Geodels/gospl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Geodels/gospl/context:python) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Total alerts](https://img.shields.io/lgtm/alerts/g/Geodels/gospl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Geodels/gospl/alerts/) |

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02804/status.svg)](https://doi.org/10.21105/joss.02804)




## Recommended installation


The easiest way to set up a full-stack scientific Python deployment is to use a Python distribution. This is an installation of Python with a set of curated packages which are guaranteed to work together. 

To start using the `gospl` recipies presented in this documentation, we recommend the **[Anaconda Python Distribution](https://www.anaconda.com/download/)**. Follow the previous link to obtain a one-click installer for Linux, Mac, or Windows. (Make sure you select the **Python 3** installer) In addition to the packages themselves, Anaconda includes a graphical utility to help manage any packages you may want to install which are not already included in the default inclusion list.

## Installation & dependencies (anaconda)

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


### Constrained/unconstrained models


### Post-processing
