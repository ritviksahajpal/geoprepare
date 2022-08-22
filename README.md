# geoprepare


[![image](https://img.shields.io/pypi/v/geoprepare.svg)](https://pypi.python.org/pypi/geoprepare)


**A Python package to prepare (download, extract, process input data) for GEOCIF and related models**


-   Free software: MIT license
-   Documentation: https://ritviksahajpal.github.io/geoprepare

## Installation
### Using PyPi (default)
```python
pip install --upgrade geoprepare
```
### Using Github repository (for development)
```python
pip install --upgrade --no-deps --force-reinstall git+https://github.com/ritviksahajpal/geoprepare.git
```

## Usage
```python
from geoprepare import geoprepare, geoextract

# Provide full path to the configuration files
# Download and preprocess data
geoprepare.run(['PATH_TO_geoprepare.txt'])

# Extract crop masks and EO variables
geoextract.run(['PATH_TO_geoprepare.txt', 'PATH_TO_geoextract.txt'])

```
Before running the code above, we need to specify the two configuration files. `geoprepare.txt` contains configuration settings for downloading and processing the input data.
`geoextract.txt` contains configuration settings for extracting crop masks and EO variables.

## Configuration files
### geoprepare.txt
`datasets`: Specify which datasets need to be downloaded and processed
`dir_base`: Path where to store the downloaded and processed files
`start_year`, `end_year`: Specify time-period for which data should be downloaded and processed
`logfile`: What directory name to use for the log files
`level`: Which level to use for [logging](https://www.loggly.com/ultimate-guide/python-logging-basics/)
`parallel_process`: Whether to use multiple CPUs
`fraction_cpus`: What fraction of available CPUs to use
```python
[DATASETS]
datasets = ['CPC', 'SOIL-MOISTURE', 'LST', 'CPC', 'AVHRR', 'AGERA5', 'CHIRPS', 'CHIRPS-GEFS']

[PATHS]
dir_base = D:\
dir_input = ${dir_base}/input
dir_log = ${dir_input}/log
dir_interim = ${dir_input}/interim
dir_download = ${dir_input}/download
dir_output = ${dir_base}/output
dir_global_datasets = ${dir_input}/global_datasets

[AGERA5]
start_year = 2022

[AVHRR]
data_dir = https://www.ncei.noaa.gov/data/avhrr-land-normalized-difference-vegetation-index/access

[CHIRPS]
fill_value = -2147483648
prelim = /pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/
final = /pub/org/chc/products/CHIRPS-2.0/global_daily/tifs/p05/
start_year = 2022

[CHIRPS-GEFS]
fill_value = -2147483648
data_dir = /pub/org/chc/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/15day/precip_mean/

[CPC]
data_dir = ftp://ftp.cdc.noaa.gov/Datasets

[ESI]
data_dir = https://gis1.servirglobal.net//data//esi//

[FLDAS]

[LST]
num_update_days = 7

[NDVI]
product = MOD09CMG
vi = ndvi
scale_glam = False
scale_mark = True
print_missing = False

[SOIL-MOISTURE]
data_dir = https://gimms.gsfc.nasa.gov/SMOS/SMAP/L03/

[LOGGING]
level = ERROR

[DEFAULT]
logfile = log
parallel_process = False
fraction_cpus = 0.5
start_year = 2022
end_year = 2022
```

### geoextract.txt
`countries`: List of countries to process
`forecast_seasons`: List of seasons to process
`mask`: Name of file to use as a mask for cropland/croptype
`redo`: Redo the processing for all days (`True`) or only days with new data (`False`)
`threshold`: Use a `threshold` value (`True`) or a `percentile` (`False`) on the cropland/croptype mask
`floor`: Value below which to set the mask to 0
`ceil`: Value above which to set the mask to 1
`eo_model`: List of datasets to extract from
```python
[kenya]
category = EWCM
calendar_file = EWCM_2021-6-17.xlsx
shp_boundary = EWCM_Level_1.shp
crops = ['mz', 'sr', 'ml', 'rc', 'ww', 'tf']
use_cropland_mask = True

[ww]
mask = cropland_v9.tif

[mz]
mask = cropland_v9.tif

[sb]
mask = cropland_v9.tif

[rc]
mask = cropland_v9.tif

[tf]
mask = cropland_v9.tif

[sr]
mask = cropland_v9.tif

[ml]
mask = cropland_v9.tif

[DEFAULT]
redo = False
threshold = True
floor = 20
ceil = 90
countries = ['kenya']
forecast_seasons = [2022]
mask = cropland_v9.tif
eo_model = ['ndvi', 'cpc_tmax', 'cpc_tmin', 'cpc_precip', 'esi_4wk', 'soil_moisture_as1', 'soil_moisture_as2']
```

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
