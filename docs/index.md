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
datasets = ['CHIRPS']<br/>
dir_base =<br/>

[DEFAULT]<br/>
logfile = log<br/>
parallel_process = True<br/>
fraction_cpus = 0.5<br/>
start_year = 1982<br/>
end_year = 2022<br/>

## Features

### CHIRPS
- Download preliminary and final CHIRPS data

### CHIRPS-GEFS
- Download preliminary and final CHIRPS-GEFS data

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
