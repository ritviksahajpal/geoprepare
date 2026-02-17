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

### 1. Download data (`geodownload`)

Downloads and preprocesses global EO datasets. Only requires `geobase.txt`. The `[DATASETS]` section controls which datasets are downloaded. Each dataset is processed to global 0.05° TIF files in `dir_intermed`.

```python
from geoprepare import geodownload

geodownload.run([r"PATH_TO_geobase.txt"])
```

**What it does:**
- Iterates over datasets listed in `[DATASETS] datasets`
- Downloads raw data from each source to `dir_download`
- Processes to standardized global TIF files in `dir_intermed`
- Automatically skips files that already exist (re-downloads only the current year if before March 1st)

**AEF dataset:** When `AEF` is in the datasets list, geodownload reads `countries` from `geoextract.txt` (must be in the same directory as `geobase.txt`) to determine which geographic tiles to download.

### 2. Validate downloads (`geocheck`)

Checks that all expected TIF files exist in `dir_intermed` and are non-empty. Writes a timestamped report to `dir_logs/check/`.

```python
from geoprepare import geocheck

# Basic check (file existence only)
geocheck.run([r"PATH_TO_geobase.txt"])

# With GDAL validation (slower, checks file readability)
geocheck.run([r"PATH_TO_geobase.txt"], gdal_check=True)
```

Supported datasets for checking: AEF, CHIRPS, NDVI, VIIRS, ESI, CPC, LST, NSIDC.

### 3. Extract crop masks and EO data (`geoextract`)

Extracts EO variable statistics (mean, median, etc.) for each admin region, crop, and growing season. Requires both `geobase.txt` and `geoextract.txt`.

```python
from geoprepare import geoextract

geoextract.run([r"PATH_TO_geobase.txt", r"PATH_TO_geoextract.txt"])
```

**What it does:**
- For each country in `[DEFAULT] countries`:
  - Reads crop mask, admin boundary shapefile, and crop calendar
  - For each crop/scale/season combination, extracts EO statistics per admin region
  - Writes per-region CSV files to `dir_output/{project_name}/crop_t{threshold}/{country}/{scale}/{crop}/{eo_var}/`
- Generates EWCM region-assignment plots in `dir_output/{project_name}/region_plots/`

**Key config values used:**
- `[DEFAULT] countries` - which countries to process
- `[PROJECT] project_name` - output subdirectory name (e.g. `FEWSNET`)
- Per-country sections - `category`, `scales`, `crops`, `growing_seasons`, `shp_boundary`, `calendar_file`
- `[EWCM]` / `[AMIS]` - `eo_model` (list of EO variables to extract), `calendar_file`

### 4. Merge extracted data (`geomerge`)

Merges per-region/year EO CSV files into a single CSV per country-crop-season combination. Adds crop calendar info, harvest season assignment, and region-to-EWCM-region mapping.

```python
from geoprepare import geomerge

geomerge.run([r"PATH_TO_geobase.txt", r"PATH_TO_geoextract.txt"])
```

**What it does:**
- For each country/scale/crop/season combination:
  - Merges all EO variable CSV files on `(country, region, region_id, year, doy)`
  - Adds datetime, crop calendar stages, harvest season assignment
  - Adds hemisphere/zone info and average temperature
  - Scales NDVI values: `(ndvi - 50) / 200`
  - Writes output to `dir_output/{project_name}/crop_t{threshold}/{country}/{country}_{crop}_s{season}.csv`
- Supports parallel execution via `[PROJECT] parallel_merge = True`

## Configuration files

geoprepare uses INI-style configuration files parsed with Python's `ConfigParser`. When multiple files are passed, they are read in order and **later files override earlier ones** for duplicate keys.

Three configuration files are used:
- **`geobase.txt`** - Paths, dataset settings, logging (required by all stages)
- **`geoextract.txt`** - Per-country extraction settings, crop masks, calendars (required by extract/merge)
- **`geocif.txt`** - geocif-specific settings, ML configs (optional, only needed for geocif pipeline)

### geobase.txt

Defines directory structure, dataset-specific settings, and global defaults.

> **NOTE:** `dir_base` needs to be changed to your specific directory structure

```ini
[DATASETS]
datasets = ['CPC']
; Available: 'NDVI', 'AGERA5', 'CHIRPS', 'CPC', 'CHIRPS-GEFS', 'ESI', 'NSIDC',
;            'FLDAS', 'VHI', 'VIIRS', 'AVHRR', 'LST', 'SOIL-MOISTURE', 'FPAR'

[PATHS]
dir_base = /gpfs/data1/cmongp1/GEO

dir_inputs = ${dir_base}/inputs
dir_logs = ${dir_base}/logs
dir_download = ${dir_inputs}/download
dir_intermed = ${dir_inputs}/intermed
dir_metadata = ${dir_inputs}/metadata

dir_boundary_files = ${dir_metadata}/boundary_files
dir_crop_calendars = ${dir_metadata}/crop_calendars
dir_crop_masks = ${dir_metadata}/crop_masks
dir_images = ${dir_metadata}/images
dir_production_statistics = ${dir_metadata}/production_statistics

dir_output = ${dir_base}/outputs

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
print_missing = False

[VIIRS]
product = VNP09CMG
vi = ndvi
scale_glam = False
scale_mark = True
print_missing = False

[LOGGING]
level = ERROR

[DEFAULT]
logfile = log
parallel_process = False
fraction_cpus = 0.35
start_year = 2001
end_year = 2026
```

### geoextract.txt

Defines per-country extraction settings, crop masks, calendar files, and EO variables.

> **NOTE:** For each country add a new section. Countries are organized into two categories: **AMIS** (global monitoring) and **EWCM** (early warning crop monitor).

**Key settings:**
- `countries`: List of countries to process (in `[DEFAULT]`)
- `category`: `AMIS` or `EWCM` - determines calendar file and EO model
- `scales`: Admin level for extraction (`admin_1` or `admin_2`)
- `crops`: List of crops to process (full names: `maize`, `sorghum`, `millet`, `rice`, `winter_wheat`, `spring_wheat`, `teff`, `soybean`)
- `growing_seasons`: List of seasons (`[1]` for primary, `[1, 2]` for primary + secondary)
- `shp_boundary`: Shapefile for admin boundaries
- `calendar_file`: Crop calendar Excel file
- `mask`: Cropland/crop-type mask filename
- `threshold` / `floor` / `ceil`: Mask thresholding settings
- `eo_model`: List of EO datasets to extract
- `redo`: Redo processing for all days (`True`) or only new data (`False`)

```ini
;;; AMIS countries ;;;
[argentina]
crops = ['soybean', 'winter_wheat', 'maize']

[brazil]
crops = ['maize', 'soybean', 'winter_wheat', 'rice']

[india]
crops = ['rice', 'maize', 'winter_wheat', 'soybean']

; ... (additional AMIS countries)

;;; EWCM countries ;;;
[kenya]
category = EWCM
scales = ['admin_1']
growing_seasons = [1, 2]
use_cropland_mask = True
shp_boundary = adm_shapefile.gpkg
calendar_file = EWCM_2025-04-21.xlsx
crops = ['maize']

[zambia]
category = EWCM
scales = ['admin_2']
growing_seasons = [1]
use_cropland_mask = True
shp_boundary = adm_shapefile.gpkg
calendar_file = EWCM_2025-04-21.xlsx
crops = ['maize']

; ... (additional EWCM countries)

;;; Crop masks ;;;
[maize]
mask = Percent_Maize.tif

[winter_wheat]
mask = Percent_Winter_Wheat.tif

[rice]
mask = Percent_Rice.tif

[soybean]
mask = Percent_Soybean.tif

; ... (additional crops)

;;; Calendar categories ;;;
[EWCM]
calendar_file = EWCM_2025-04-21.xlsx
eo_model = ['nsidc_surface', 'nsidc_rootzone', 'ndvi', 'cpc_tmax', 'cpc_tmin', 'chirps', 'chirps_gefs', 'esi_4wk']

[AMIS]
calendar_file = AMISCM_2025-04-21.xlsx

[DEFAULT]
method = JRC
redo = True
threshold = True
floor = 20
ceil = 90
countries = ["kenya"]
crops = ['maize']
category = AMIS
scales = ['admin_1']
growing_seasons = [1]
mask = cropland_v9.tif
use_cropland_mask = False
shp_boundary = gaul1_asap_v04.shp
statistics_file = statistics.csv
zone_file = countries.csv
eo_model = ['aef', 'nsidc_surface', 'nsidc_rootzone', 'ndvi', 'cpc_tmax', 'cpc_tmin', 'chirps', 'chirps_gefs', 'esi_4wk']

[PROJECT]
project_name = FEWSNET
parallel_extract = True
parallel_merge = False
```

### geocif.txt (optional)

Only needed when running the geocif pipeline. Adds per-country geocif settings, ML model configuration, and overrides `[DEFAULT]` values like `countries`, `method`, and `project_name`.

**Key overrides when geocif.txt is loaded:**
- `countries` changes from geoextract.txt's list to geocif's list
- `method` changes from `JRC` to `monthly_r`
- `project_name` in `[DEFAULT]` becomes `geocif` (but `[PROJECT] project_name` stays `FEWSNET`)

```ini
[AGMET]
eo_plot = ['ndvi', 'cpc_tmax', 'cpc_tmin', 'chirps', 'esi_4wk', 'nsidc_surface', 'nsidc_rootzone']

;;; Per-country geocif settings ;;;
[zambia]
crops = ['maize']
admin_zone = admin_2
boundary_file = adm_shapefile.gpkg

[malawi]
crops = ['maize']
admin_zone = admin_2
boundary_file = adm_shapefile.gpkg

; ... (additional countries)

;;; ML model settings ;;;
[ML]
model_type = REGRESSION
target = Yield (tn per ha)
panel_model = True
lag_years = 3
; ... (see geocif.txt for full ML configuration)

[DEFAULT]
data_source = harvest
method = monthly_r
project_name = geocif
countries = ["zambia", "malawi", "south_africa", "madagascar", "mozambique", "angola", "zimbabwe"]
crops = ['maize']
admin_zone = admin_1
models = ['catboost']
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
This project was supported by was supported by NASA Applied Sciences Grant No. 80NSSC17K0625 through the NASA Harvest Consortium,
and the NASA Acres Consortium under NASA Grant #80NSSC23M0034
