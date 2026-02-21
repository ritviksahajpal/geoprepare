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

```bash
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
```bash
pip install git+https://github.com/ritviksahajpal/octvi.git
```

Downloading from the NASA distributed archives (DAACs) requires a personal app key. Users must
configure the module using a new console script, `octviconfig`. After installation, run `octviconfig`
in your command prompt to prompt the input of your personal app key. Information on obtaining app keys
can be found [here](https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#tokens)


### Using PyPi (default)
```bash
pip install --upgrade geoprepare
```

### Using Github repository (for development)
```bash
pip install --upgrade --no-deps --force-reinstall git+https://github.com/ritviksahajpal/geoprepare.git
```

### Local installation
Navigate to the directory containing `pyproject.toml` and run the following command:
```bash
pip install .
```

For development (editable install):
```bash
pip install -e ".[dev]"
```

## Pipeline

geoprepare follows a three-stage pipeline:

1. **Download** (`geodownload`) - Download and preprocess global EO datasets to `dir_download` and `dir_intermed`
2. **Extract** (`geoextract`) - Extract EO variable statistics per admin region to `dir_output`
3. **Merge** (`geomerge`) - Merge extracted EO files into per-country/crop CSV files for ML models and AgMet graphics

Additional utilities:
- **Check** (`geocheck`) - Validate that expected TIF files exist in `dir_intermed` after download
- **Diagnostics** (`diagnostics`) - Count and summarize files in the data directories

## Usage

```python
config_dir = "/path/to/config"  # full path to your config directory

cfg_geoprepare = [f"{config_dir}/geobase.txt", f"{config_dir}/countries.txt", f"{config_dir}/crops.txt", f"{config_dir}/geoextract.txt"]
```

### 1. Download data (`geodownload`)

Downloads and preprocesses global EO datasets. Only requires `geobase.txt`. The `[DATASETS]` section controls which datasets are downloaded. Each dataset is processed to global 0.05° TIF files in `dir_intermed`.

```python
from geoprepare import geodownload
geodownload.run([f"{config_dir}/geobase.txt"])
```

### 2. Validate downloads (`geocheck`)

Checks that all expected TIF files exist in `dir_intermed` and are non-empty. Writes a timestamped report to `dir_logs/check/`.

```python
from geoprepare import geocheck
geocheck.run([f"{config_dir}/geobase.txt"])
```

### 3. Extract crop masks and EO data (`geoextract`)

Extracts EO variable statistics (mean, median, etc.) for each admin region, crop, and growing season.

```python
from geoprepare import geoextract
geoextract.run(cfg_geoprepare)
```

### 4. Merge extracted data (`geomerge`)

Merges per-region/year EO CSV files into a single CSV per country-crop-season combination.

```python
from geoprepare import geomerge
geomerge.run(cfg_geoprepare)
```

## Config files

| File | Purpose | Used by |
|------|---------|---------|
| [`geobase.txt`](#geobasetxt) | Paths, dataset settings, boundary file column mappings, logging | both |
| [`countries.txt`](#countriestxt) | Per-country config (boundary files, admin levels, seasons, crops) | both |
| [`crops.txt`](#cropstxt) | Crop masks, calendar categories (EWCM, AMIS), EO model variables | both |
| [`geoextract.txt`](#geoextracttxt) | Extraction-only settings (method, threshold, parallelism) | geoprepare |
| [`geocif.txt`](#geociftxt) | Indices/ML/agmet settings, country overrides, runtime selections | geocif |

**Order matters:** Config files are loaded left-to-right. When the same key appears in multiple files, the last file wins. The tool-specific file (`geoextract.txt` or `geocif.txt`) must be last so its `[DEFAULT]` values (countries, method, etc.) override the shared defaults in `countries.txt`.

```python
config_dir = "/path/to/config"  # full path to your config directory

cfg_geoprepare = [f"{config_dir}/geobase.txt", f"{config_dir}/countries.txt", f"{config_dir}/crops.txt", f"{config_dir}/geoextract.txt"]
cfg_geocif = [f"{config_dir}/geobase.txt", f"{config_dir}/countries.txt", f"{config_dir}/crops.txt", f"{config_dir}/geocif.txt"]
```

## Config file documentation

### geobase.txt

Shared paths, dataset settings, boundary file column mappings, and logging. All directory paths are derived from `dir_base`.

```ini
[DATASETS]
datasets = ['CHIRPS', 'CPC', 'NDVI', 'ESI', 'NSIDC', 'AEF']

[PATHS]
dir_base = /gpfs/data1/cmongp1/GEO

dir_inputs = ${dir_base}/inputs
dir_logs = ${dir_base}/logs
dir_download = ${dir_inputs}/download
dir_intermed = ${dir_inputs}/intermed
dir_metadata = ${dir_inputs}/metadata
dir_condition = ${dir_inputs}/crop_condition
dir_crop_inputs = ${dir_condition}/crop_t20

dir_boundary_files = ${dir_metadata}/boundary_files
dir_crop_calendars = ${dir_metadata}/crop_calendars
dir_crop_masks = ${dir_metadata}/crop_masks
dir_images = ${dir_metadata}/images
dir_production_statistics = ${dir_metadata}/production_statistics

dir_output = ${dir_base}/outputs

; --- Per-dataset settings ---

[AEF]
; AlphaEarth Foundations satellite embeddings (2018-2024, 64 channels, 10m)
; Source: https://source.coop/tge-labs/aef  |  License: CC-BY 4.0
; Countries are read from geoextract.txt [DEFAULT] countries
buffer = 0.5
download_vrt = True
start_year = 2018
end_year = 2024

[AGERA5]
variables = ['Precipitation_Flux', 'Temperature_Air_2m_Max_24h', 'Temperature_Air_2m_Min_24h']

[CHIRPS]
fill_value = -2147483648
; CHIRPS version: 'v2' for CHIRPS-2.0 or 'v3' for CHIRPS-3.0
version = v3
; Disaggregation method for v3 only: 'sat' (IMERG) or 'rnl' (ERA5)
; - 'sat': Uses NASA IMERG Late V07 for daily downscaling (available from 1998, 0.1° resolution)
; - 'rnl': Uses ECMWF ERA5 for daily downscaling (full time coverage, 0.25° resolution)
; Note: Prelim data is only available with 'sat' due to ERA5 latency (5-6 days)
disagg = sat

[CHIRPS-GEFS]
fill_value = -2147483648
data_dir = /pub/org/chc/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/15day/precip_mean/

[CPC]
data_dir = ftp://ftp.cdc.noaa.gov/Datasets

[ESI]
data_dir = https://gis1.servirglobal.net//data//esi//
list_products = ['4wk', '12wk']

[FLDAS]
use_spear = False
data_types = ['forecast']
variables = ['SoilMoist_tavg', 'TotalPrecip_tavg', 'Tair_tavg', 'Evap_tavg', 'TWS_tavg']
leads = [0, 1, 2, 3, 4, 5]
compute_anomalies = False

[NDVI]
product = MOD09CMG
vi = ndvi
scale_glam = False
scale_mark = True

[VIIRS]
product = VNP09CMG
vi = ndvi
scale_glam = False
scale_mark = True

[NSIDC]

[VHI]
data_historic = https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/VHP_4km_GeoTiff/
data_current = https://www.star.nesdis.noaa.gov/pub/corp/scsb/wguo/data/Blended_VH_4km/geo_TIFF/

; --- Boundary file column mappings ---
; Section name = filename stem (without extension)
; Maps source shapefile columns to standard internal names:
;   adm0_col  -> ADM0_NAME (country)
;   adm1_col  -> ADM1_NAME (admin level 1)
;   adm2_col  -> ADM2_NAME (admin level 2, optional)
;   id_col    -> ADM_ID    (unique feature ID)

[adm_shapefile]
adm0_col = ADMIN0
adm1_col = ADMIN1
adm2_col = ADMIN2
id_col = FNID

[gaul1_asap_v04]
adm0_col = name0
adm1_col = name1
id_col = asap1_id

[EWCM_Level_1]
adm0_col = ADM0_NAME
adm1_col = ADM1_NAME
id_col = num_ID

; Add more [boundary_stem] sections as needed for other shapefiles

[LOGGING]
level = ERROR

[POOCH]
; URL to download metadata.zip (boundary files, crop masks, calendars, etc.)
; NOTE: Set this to your own hosted URL (e.g. Dropbox, S3, etc.)
url = <your_metadata_zip_url>
enabled = True

[DEFAULT]
logfile = log
parallel_process = False
fraction_cpus = 0.35
start_year = 2001
end_year = 2026
```

### countries.txt

Single source of truth for per-country config. Shared by both geoprepare and geocif.

```ini
[DEFAULT]
boundary_file = gaul1_asap_v04.shp
admin_level = admin_1
seasons = [1]
crops = ['maize']
category = AMIS
use_cropland_mask = False
calendar_file = crop_calendar.csv
mask = cropland_v9.tif
statistics_file = statistics.csv
zone_file = countries.csv
shp_region = GlobalCM_Regions_2025-11.shp
eo_model = ['aef', 'nsidc_surface', 'nsidc_rootzone', 'ndvi', 'cpc_tmax', 'cpc_tmin', 'chirps', 'chirps_gefs', 'esi_4wk']
annotate_regions = False

;;; AMIS countries (inherit from DEFAULT, override crops if needed) ;;;
[argentina]
crops = ['soybean', 'winter_wheat', 'maize']

[brazil]
crops = ['maize', 'soybean', 'winter_wheat', 'rice']

[india]
crops = ['rice', 'maize', 'winter_wheat', 'soybean']

[united_states_of_america]
crops = ['rice', 'maize', 'winter_wheat']

; ... (40+ AMIS countries, most inherit DEFAULT crops)

;;; EWCM countries (full per-country config) ;;;
[kenya]
category = EWCM
admin_level = admin_1
seasons = [1, 2]
use_cropland_mask = True
boundary_file = adm_shapefile.gpkg
calendar_file = EWCM_2025-04-21.xlsx
crops = ['maize']

[malawi]
category = EWCM
admin_level = admin_2
use_cropland_mask = True
boundary_file = adm_shapefile.gpkg
calendar_file = EWCM_2025-04-21.xlsx
crops = ['maize']

[ethiopia]
category = EWCM
admin_level = admin_2
use_cropland_mask = True
boundary_file = adm_shapefile.gpkg
calendar_file = EWCM_2025-04-21.xlsx
crops = ['maize', 'sorghum', 'millet', 'rice', 'winter_wheat', 'teff']

; ... (30+ EWCM countries, mostly Sub-Saharan Africa)

;;; Other countries (custom boundary files, non-standard setups) ;;;
[nepal]
crops = ['rice']
boundary_file = hermes_NPL_new_wgs_2.shp

[illinois]
admin_level = admin_3
boundary_file = illinois_counties.shp
```

### crops.txt

Crop mask filenames and calendar category definitions. Calendar categories define the EO variables and crop calendars used for each category of countries.

```ini
;;; Crop masks ;;;
[winter_wheat]
mask = Percent_Winter_Wheat.tif

[spring_wheat]
mask = Percent_Spring_Wheat.tif

[maize]
mask = Percent_Maize.tif

[soybean]
mask = Percent_Soybean.tif

[rice]
mask = Percent_Rice.tif

[teff]
mask = cropland_v9.tif

[sorghum]
mask = cropland_v9.tif

[millet]
mask = cropland_v9.tif

;;; Calendar categories ;;;
[EWCM]
use_cropland_mask = True
shp_boundary = adm_shapefile.gpkg
calendar_file = EWCM_2026-01-05.xlsx
crops = ['maize', 'sorghum', 'millet', 'rice', 'winter_wheat', 'teff']
growing_seasons = [1]
eo_model = ['aef', 'nsidc_surface', 'nsidc_rootzone', 'ndvi', 'cpc_tmax', 'cpc_tmin', 'chirps', 'chirps_gefs', 'esi_4wk']

[AMIS]
calendar_file = AMISCM_2026-01-05.xlsx
```

### geoextract.txt

Extraction-only settings for geoprepare. Loaded last so its `[DEFAULT]` overrides shared defaults.

```ini
[DEFAULT]
project_name = geocif
method = JRC
redo = False
threshold = True
floor = 20
ceil = 90
countries = ["malawi"]
forecast_seasons = [2022]

[PROJECT]
parallel_extract = True
parallel_merge = False
```

### geocif.txt

Indices, ML, and agmet settings for geocif. Country overrides go here when geocif needs different values than countries.txt (e.g., a subset of crops). Its `[DEFAULT]` section is loaded last and overrides shared defaults for geocif runs.

```ini
[AGMET]
eo_plot = ['ndvi', 'cpc_tmax', 'cpc_tmin', 'chirps', 'esi_4wk', 'nsidc_surface', 'nsidc_rootzone']
logo_harvest = harvest.png
logo_geoglam = geoglam.png

;;; Country overrides (only where geocif differs from countries.txt) ;;;
[ethiopia]
crops = ['winter_wheat']

[bangladesh]
crops = ['rice']
admin_level = admin_2
boundary_file = bangladesh.shp

[india]
crops = ['soybean', 'maize', 'rice']

[somalia]
crops = ['maize']

[ukraine]
crops = ['winter_wheat', 'maize']

;;; ML model definitions ;;;
[catboost]
ML_model = True

[linear]
ml_model = True

[analog]
ML_model = False

[median]
ML_model = False

; ... (additional models: gam, ngboost, tabpfn, desreg, cubist, etc.)

[ML]
model_type = REGRESSION
target = Yield (tn per ha)
feature_selection = BorutaPy
lag_years = 3
panel_model = True
panel_model_region = Country
median_years = 5
lag_yield_as_feature = True
run_latest_time_period = True
run_every_time_period = 3
cat_features = ["Harvest Year", "Region_ID", "Region"]
loocv_var = Harvest Year

[LOGGING]
log_level = INFO

[DEFAULT]
data_source = harvest
method = monthly_r
project_name = geocif
countries = ["kenya"]
crops = ['maize']
admin_level = admin_1
models = ['catboost']
seasons = [1]
threshold = True
floor = 20
input_file_path = ${PATHS:dir_crop_inputs}/processed
```

## Supported datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| AEF | AlphaEarth Foundations satellite embeddings (64-band, 10m) | [source.coop](https://source.coop/tge-labs/aef) |
| AGERA5 | Agrometeorological indicators (precipitation, temperature) | [CDS](https://cds.climate.copernicus.eu) |
| CHIRPS | Rainfall estimates (v2 and v3) | [CHC](https://www.chc.ucsb.edu/data/chirps) |
| CHIRPS-GEFS | 15-day precipitation forecasts | CHC |
| CPC | Temperature (Tmax, Tmin) | NOAA CPC |
| ESI | Evaporative Stress Index (4-week, 12-week) | SERVIR |
| FLDAS | Land surface model outputs (soil moisture, precip, temp) | NASA |
| NDVI | Vegetation index from MODIS (MOD09CMG) | NASA |
| VIIRS | Vegetation index from VIIRS (VNP09CMG) | NASA |
| NSIDC | Soil moisture (surface, rootzone) | NSIDC |
| VHI | Vegetation Health Index | NOAA STAR |
| LST | Land Surface Temperature | NASA |
| AVHRR | Long-term NDVI | NOAA NCEI |
| FPAR | Fraction of Absorbed Photosynthetically Active Radiation | JRC |
| SOIL-MOISTURE | SMAP soil moisture | NASA |

## Upload package to PyPI

Navigate to the **root of the geoprepare repository** (the directory containing `pyproject.toml`):
```bash
cd /path/to/geoprepare
```

### Step 1: Update version
Use `bump2version` to update the version in both `pyproject.toml` and `geoprepare/__init__.py`:

**Using uv:**
```bash
uvx bump2version patch --current-version X.X.X --new-version X.X.Y pyproject.toml geoprepare/__init__.py
```

**Using pip:**
```bash
pip install bump2version
bump2version patch --current-version X.X.X --new-version X.X.Y pyproject.toml geoprepare/__init__.py
```

Or manually edit the version in `pyproject.toml` and `geoprepare/__init__.py`.

### Step 2: Clean old builds

**Linux/macOS:**
```bash
rm -rf dist/ build/ *.egg-info/
```

**Windows (Command Prompt):**
```cmd
rmdir /s /q dist build geoprepare.egg-info
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force dist/, build/, *.egg-info/ -ErrorAction SilentlyContinue
```

### Step 3: Build and upload

**Using uv (Linux/macOS):**
```bash
uv build
uvx twine check dist/*
uvx twine upload dist/geoprepare-X.X.X*
```

**Using uv (Windows):**
```cmd
uv build
uvx twine check dist\geoprepare-X.X.X.tar.gz dist\geoprepare-X.X.X-py3-none-any.whl
uvx twine upload dist\geoprepare-X.X.X.tar.gz dist\geoprepare-X.X.X-py3-none-any.whl
```

**Using pip:**
```bash
pip install build twine
python -m build
twine check dist/*
twine upload dist/geoprepare-X.X.X*
```

Replace `X.X.X` with your current version and `X.X.Y` with the new version.

### Optional: Configure PyPI credentials
To avoid entering credentials each time, create a `~/.pypirc` file (Linux/macOS) or `%USERPROFILE%\.pypirc` (Windows):
```ini
[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE
```

## Credits

This project was supported by NASA Applied Sciences Grant No. 80NSSC17K0625 through the NASA Harvest Consortium, and the NASA Acres Consortium under NASA Grant #80NSSC23M0034.
