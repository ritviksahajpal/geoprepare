# geoprepare


[![image](https://img.shields.io/pypi/v/geoprepare.svg)](https://pypi.python.org/pypi/geoprepare)


**A Python package to prepare (download, extract, process input data) for GEOCIF and related models**


-   Free software: MIT license
-   Documentation: https://ritviksahajpal.github.io/geoprepare

## Installation
`pip install --upgrade --no-deps --force-reinstall git+https://github.com/ritviksahajpal/geoprepare.git`

## Usage
`import geoprepare.geoprepare as gprp`<br/>
`gprp.run(PATH_TO_CONFIG_FILE)`

A sample config.txt file can be found in the `geoprepare` folder

## Config file
[DATASETS]<br/>
datasets = ['NDVI', 'SOIL-MOISTURE', 'LST', 'CPC', 'AVHRR', 'AGERA5', 'CHIRPS', 'CHIRPS-GEFS']<br/>
dir_base = D:\<br/>
dir_input = ${dir_base}/input<br/>
dir_log = ${dir_input}/log<br/>
dir_interim = ${dir_input}/interim<br/>
dir_download = ${dir_input}/download<br/>
dir_output = ${dir_base}/output<br/>

[AGERA5]<br/>
start_year = 2022<br/>

[AVHRR]<br/>
data_dir = https://www.ncei.noaa.gov/data/avhrr-land-normalized-difference-vegetation-index/access<br/>
<br/>
[CHIRPS]<br/>
fill_value = -2147483648<br/>
prelim = /pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/<br/>
final = /pub/org/chc/products/CHIRPS-2.0/global_daily/tifs/p05/<br/>
start_year = 2022<br/>
<br/>
[CHIRPS-GEFS]<br/>
fill_value = -2147483648<br/>
data_dir = /pub/org/chc/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/15day/precip_mean/<br/>
<br/>
[CPC]<br/>
data_dir = ftp://ftp.cdc.noaa.gov/Datasets<br/>
<br/>
[ESI]<br/>
data_dir = https://gis1.servirgloba<br/>l.net//<br/>data//esi//<br/>
<br/>
[FLDAS]<br/>
<br/>
[LST]<br/>
num_update_days = 7<br/>
<br/>
[NDVI]<br/>
product = MOD09CMG<br/>
vi = ndvi<br/>
scale_glam = False<br/>
scale_mark = True<br/>
print_missing = False<br/>
<br/>
[SOIL-MOISTURE]<br/>
data_dir = https://gimms.gsfc.nasa.gov/SMOS/SMAP/L03/<br/>
<br/>
[DEFAULT]<br/>
logfile = log<br/>
parallel_process = True<br/>
fraction_cpus = 0.5<br/>
start_year = 2022<br/>
end_year = 2022<br/>
## Features

### CHIRPS
- Download preliminary and final CHIRPS data

### CHIRPS-GEFS
- Download preliminary and final CHIRPS-GEFS data

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
