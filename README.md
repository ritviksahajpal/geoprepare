# geoprepare


[![image](https://img.shields.io/pypi/v/geoprepare.svg)](https://pypi.python.org/pypi/geoprepare)


**A Python package to prepare (download, extract, process input data) for GEOCIF and related models**


-   Free software: MIT license
-   Documentation: https://ritviksahajpal.github.io/geoprepare

## Installation
```python
pip install --upgrade --no-deps --force-reinstall git+https://github.com/ritviksahajpal/geoprepare.git
```

## Usage
```python
from geoprepare import geoprepare, geoextract

# Provide full path to the configuration files
# Download and preprocess data
geoprepare.run(['PATH_TO_geoprepare.txt', 'PATH_TO_geoextract.txt'])

# Extract crop masks and EO variables
geoextract.run(['PATH_TO_geoprepare.txt', 'PATH_TO_geoextract.txt'])

```
These files can be found in the `geoprepare` folder and can be adapted to your machine

## Config files
### geoprepare.txt
Contains information on:
1. Path where to store the downloaded and processed files: `dir_base`
2. Specify which datasets need to be downloaded and processed: `datasets`
3. Specify time-period for which data should be downloaded and processed: `start_year`, `end_year`
4. What fraction of available CPUs to use: `fraction_cpus`
5. What directory name to use for the log files: `logfile`
6. Whether to use multiple CPUs: `parallel_process`


### geoextract.txt
1. List of countries to process: `countries`
2. List of seasons to process: `forecast_seasons`

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
