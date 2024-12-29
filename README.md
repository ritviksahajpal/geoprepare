# geoprepare


[![image](https://img.shields.io/pypi/v/geoprepare.svg)](https://pypi.python.org/pypi/geoprepare)


**A Python package to prepare (download, extract, process input data) for GEOCIF and related models**


-   Free software: MIT license
-   Documentation: https://ritviksahajpal.github.io/geoprepare

## Installation
> **Note:** The instructions below have only been tested on a Linux system

### Install Anaconda
We recommend that you use the conda package manager to install the `geoprepare` library and all its
dependencies. If you do not have it installed already, you can get it from the [Anaconda distribution](https://www.anaconda.com/download#downloads)

### Using the CDS API
If you intend to download AgERA5 data, you will need to install the CDS API.
You can do this by following the instructions [here](https://cds.climate.copernicus.eu/api-how-to)

### Create a new conda environment (optional but highly recommended)
`geoprepare` requires multiple Python GIS packages including `gdal` and `rasterio`. These packages are not always easy
to install. To make the process easier, you can optionally create a new environment using the
following commands, specify the python version you have on your machine (python >= 3.9 is recommended). we use the `pygis` library
to install multiple Python GIS packages including `gdal` and `rasterio`.

```python
conda create --name <name_of_environment> python=3.x
conda activate <name_of_environment>
conda install -c conda-forge mamba
mamba install -c conda-forge gdal
mamba install -c conda-forge rasterio
mamba install -c conda-forge xarray
mamba install -c conda-forge rioxarray
mamba install -c conda-forge pyresample
mamba install -c conda-forge cdsapi
mamba install -c conda-forge pygis
pip install wget
pip install pyl4c
```

Install the octvi package to download MODIS data
```python
pip install git+https://github.com/ritviksahajpal/octvi.git
```

Downloading from the NASA distributed archives (DAACs) requires a personal app key. Users must
configure the module using a new console script, `octviconfig`. After installation, run `octviconfig`
in your command prompt to prompt the input of your personal app key. Information on obtaining app keys
can be found [here](https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#tokens)


### Using PyPi (default)
```python
pip install --upgrade geoprepare
```

### Using Github repository (for development)
```python
pip install --upgrade --no-deps --force-reinstall git+https://github.com/ritviksahajpal/geoprepare.git
```

### Local installation
Navigate to the directory containing `setup.py` and run the following command:
```python
pip install .
```

## Usage
* Execute the following code to download the data

```python
from geoprepare import geodownload

# Provide full path to the configuration files
# Download and preprocess data
geodownload.run([r"PATH_TO_geobase.txt"])
```

* Execute the following code to extract crop masks and EO data
```python
from geoprepare import geoextract

# Extract crop masks and EO variables
geoextract.run([r"PATH_TO_geobase.txt", r"PATH_TO_geoextract.txt"])
```

* Execute the following code to prepare the data for the crop yield ML model and AgMet graphics
```python
from geoprepare import geomerge

# Merge EO files into one, this is needed to create AgMet graphics and to run the crop yield model
geomerge.run([r"PATH_TO_geobase.txt", r"PATH_TO_geoextract.txt"])
```


Before running the code above, we need to specify the two configuration files:
* `geobase.txt` contains configuration settings for downloading and processing the input data.
* `geoextract.txt` contains configuration settings for extracting crop masks and EO variables.

## Configuration files
### geobase.txt
> **NOTE:** `dir_base` needs to be changed to your specific directory structure
* `datasets`: Specify which datasets need to be downloaded and processed
* `dir_base`: Path where to store the downloaded and processed files
* `start_year`, `end_year`: Specify time-period for which data should be downloaded and processed
* `logfile`: What directory name to use for the log files
* `level`: Which level to use for [logging](https://www.loggly.com/ultimate-guide/python-logging-basics/)
* `parallel_process`: Whether to use multiple CPUs
* `fraction_cpus`: What fraction of available CPUs to use
```python
[DATASETS]
datasets = ['NDVI', 'AGERA5', 'CHIRPS', 'CPC', 'CHIRPS-GEFS', 'NSIDC']

[PATHS]
dir_base = /gpfs/data1/cmongp1/GEOGLAM
dir_input = ${dir_base}/Input
dir_log = ${dir_base}/log
dir_interim = ${dir_input}/intermed
dir_download = ${dir_input}/download
dir_output = ${dir_base}/Output
dir_global_datasets = ${dir_input}/Global_Datasets
dir_metadata = ${dir_input}/metadata
dir_masks = ${dir_global_datasets}/masks
dir_regions = ${dir_global_datasets}/regions
dir_regions_shp = ${dir_regions}/shps
dir_crop_masks = ${dir_input}/crop_masks
dir_models = ${dir_input}/models

[AGERA5]
variables = ['Precipitation_Flux', 'Temperature_Air_2m_Max_24h', 'Temperature_Air_2m_Min_24h']

[AVHRR]
data_dir = https://www.ncei.noaa.gov/data/avhrr-land-normalized-difference-vegetation-index/access

[CHIRPS]
fill_value = -2147483648
prelim = /pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/
final = /pub/org/chc/products/CHIRPS-2.0/global_daily/tifs/p05/

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

[VHI]
data_historic = https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/VHP_4km_GeoTiff/
data_current = https://www.star.nesdis.noaa.gov/pub/corp/scsb/wguo/data/Blended_VH_4km/geo_TIFF/

[NDVI]
product = MOD09CMG
vi = ndvi
scale_glam = False
scale_mark = True
print_missing = False

[VIIRS]
product = VNP09CMG
vi = ndvi
scale_glam = False
scale_mark = True
print_missing = False

[NSIDC]

[SOIL-MOISTURE]
data_dir = https://gimms.gsfc.nasa.gov/SMOS/SMAP/L03/

[FPAR]
data_url = https://agricultural-production-hotspots.ec.europa.eu/data/indicators_fpar/fpar/

[LOGGING]
level = ERROR

[DEFAULT]
logfile = log
parallel_process = False
fraction_cpus = 0.75
start_year = 2001
end_year = 2024
```

### geoextract.txt
> **NOTE:** For each country add a new section to this file, using `kenya` as an example
* `countries`: List of countries to process
* `forecast_seasons`: List of seasons to process
* `mask`: Name of file to use as a mask for cropland/croptype
* `redo`: Redo the processing for all days (`True`) or only days with new data (`False`)
* `threshold`: Use a `threshold` value (`True`) or a `percentile` (`False`) on the cropland/croptype mask
* `floor`: Value below which to set the mask to 0
* `ceil`: Value above which to set the mask to 1
* `eo_model`: List of datasets to extract from
```python
[kenya]
category = EWCM
scales = ['admin_1']  ; can be admin_1 (state level) or admin_2 (county level)
growing_seasons = [1]  ; 1 is primary/long season, 2 is secondary/short season
crops = ['mz', 'sr', 'ml', 'rc', 'ww', 'tf']
use_cropland_mask = True

[rwanda]
category = EWCM
scales = ['admin_1']  ; can be admin_1 (state level) or admin_2 (county level)
growing_seasons = [1]  ; 1 is primary/long season, 2 is secondary/short season
crops = ['mz', 'sr', 'ml', 'rc', 'ww', 'tf']
use_cropland_mask = True

[malawi]
category = EWCM
scales = ['admin_1']  ; can be admin_1 (state level) or admin_2 (county level)
growing_seasons = [1]  ; 1 is primary/long season, 2 is secondary/short season
crops = ['mz', 'sr', 'ml', 'rc', 'ww', 'tf']
use_cropland_mask = True

[zambia]
category = EWCM
scales = ['admin_1']  ; can be admin_1 (state level) or admin_2 (county level)
growing_seasons = [1]  ; 1 is primary/long season, 2 is secondary/short season
crops = ['mz', 'sr', 'ml', 'rc', 'ww', 'tf']
use_cropland_mask = True

[united_republic_of_tanzania]
category = EWCM
scales = ['admin_1']  ; can be admin_1 (state level) or admin_2 (county level)
growing_seasons = [1]  ; 1 is primary/long season, 2 is secondary/short season
crops = ['mz', 'sr', 'ml', 'rc', 'ww', 'tf']
use_cropland_mask = True

[ww]
mask = cropland_v9.tif  ; A tif file specifying name of cropland/crop-type mask

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

[EWCM]
calendar_file = EWCM_2021-6-17.xlsx

[AMIS]
calendar_file = AMISCM_2021-6-17.xlsx

[DEFAULT]
redo = False
threshold = True
floor = 20
ceil = 90
scales = ['admin_1']
growing_seasons = [1]
countries = ['kenya']
forecast_seasons = [2022]
mask = cropland_v9.tif
shp_boundary = EWCM_Level_1.shp
statistics_file = statistics.csv
zone_file = countries.csv
calendar_file = crop_calendar.csv
eo_model = ['ndvi', 'cpc_tmax', 'cpc_tmin', 'chirps', 'chirps_gefs', 'esi_4wk', 'soil_moisture_as1', 'soil_moisture_as2']
```

## Accessing EO data using the earthaccess library
```python
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path

from geoprepare.eoaccess import eoaccess

dg = gpd.read_file(PATH_TO_SHAPEFILE, engine="pyogrio")

# Convert to CRS 4326 if not already
if dg.crs != "EPSG:4326":
    dg = dg.to_crs("EPSG:4326")

# Iterate over each row of the shapefile
for index, row in tqdm(dg.iterrows(), desc="Iterating over shapefile", total=len(dg)):
    # Get bbox from geometry of the row
    bbox = row.geometry.bounds

    obj = eoaccess.NASAEarthAccess(
        dataset=["HLSL30", "HLSS30"],
        bbox=bbox,
        temporal=(f"{row['year']}-01-01", f"{row['year']}-12-31"),
        output_dir=".",
    )

    obj.search_data()
    if obj.results:
        obj.download_parallel()

obj = eoaccess.EarthAccessProcessor(
    dataset=["HLSL30", "HLSS30"],
    input_dir=".",
    shapefile=Path(PATH_TO_SHAPEFILE),
)
obj.mosaic()
```

### Upload package to pypi
1. Update requirements.txt
2. Update version="A.B.C" in setup.py
3. Navigate to the directory containing `setup.py` and run the following command:
```python
pipreqs . --force --savepath requirements.txt
mamba env export > environment.yml
python setup.py sdist
twine upload dist/geoprepare-A.B.C.tar.gz
```

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
