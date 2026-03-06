# geoprepare


[![image](https://img.shields.io/pypi/v/geoprepare.svg)](https://pypi.python.org/pypi/geoprepare)


**A Python package to prepare (download, extract, process input data) for GEOCIF and related models**


-   Free software: MIT license
-   Documentation: https://ritviksahajpal.github.io/geoprepare

## Installation

### Install from PyPI
```bash
pip install --upgrade geoprepare
```

### Install from GitHub (development)
```bash
pip install --upgrade --no-deps --force-reinstall git+https://github.com/ritviksahajpal/geoprepare.git
```

### Local editable install
```bash
pip install -e ".[dev]"
```

### CDS API (for AgERA5)
If you intend to download AgERA5 data, install the CDS API by following the instructions [here](https://cds.climate.copernicus.eu/api-how-to).

### MODIS data (octvi)
Install the octvi package to download MODIS data:
```bash
pip install git+https://github.com/ritviksahajpal/octvi.git
```

Downloading from the NASA DAACs requires a personal app key. After installation, run `octviconfig` in your command prompt. Information on obtaining app keys can be found [here](https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#tokens).

## Pipeline

geoprepare follows a three-stage pipeline:

1. **Download** (`geodownload`) - Download and preprocess global EO datasets to `dir_download` and `dir_intermed`
2. **Extract** (`geoextract`) - Extract EO variable statistics per admin region to `dir_output`
3. **Merge** (`geomerge`) - Merge extracted EO files into per-country/crop CSV files for ML models and AgMet graphics

All datasets store files in year-specific subfolders (e.g., `dir_intermed/cpc_tmax/2024/`, `dir_download/nsidc/2025/`).

Additional utilities:
- **Move** (`geomove`) - One-time migration of existing flat directories to year-specific subfolders
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

### 2. Migrate to year subfolders (`geomove`)

Moves existing files from flat directories into year-specific subfolders. Run this once after upgrading to a version with year-subfolder support. All datasets are handled: CPC, ESI, NDVI, NSIDC, CHIRPS-GEFS, LST, Soil Moisture, AgERA5, VHI, FPAR, and AEF.

```python
from geoprepare import geomove

# Preview what would be moved (no files are changed)
geomove.run([f"{config_dir}/geobase.txt"], dry_run=True)

# Execute the migration
geomove.run([f"{config_dir}/geobase.txt"])
```

### 3. Validate downloads (`geocheck`)

Checks that all expected TIF files exist in `dir_intermed` and are non-empty. Writes a timestamped report to `dir_logs/check/`.

```python
from geoprepare import geocheck
geocheck.run([f"{config_dir}/geobase.txt"])
```

### 4. Extract crop masks and EO data (`geoextract`)

Extracts EO variable statistics (mean, median, etc.) for each admin region, crop, and growing season.

```python
from geoprepare import geoextract
geoextract.run(cfg_geoprepare)
```

### 5. Merge extracted data (`geomerge`)

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
| [`crops.txt`](#cropstxt) | Crop masks, calendar category settings (EWCM, AMIS) | both |
| [`geoextract.txt`](#geoextracttxt) | Extraction-only settings (method, threshold, parallelism) | geoprepare |
| [`geocif.txt`](#geociftxt) | Indices/ML/agmet settings, country overrides, runtime selections | geocif |

**Order matters:** Config files are loaded left-to-right. When the same key appears in multiple files, the last file wins. The tool-specific file (`geoextract.txt` or `geocif.txt`) must be last so its `[DEFAULT]` values (countries, method, etc.) override the shared defaults in `countries.txt`.

```python
config_dir = "/path/to/config"

cfg_geoprepare = [f"{config_dir}/geobase.txt", f"{config_dir}/countries.txt", f"{config_dir}/crops.txt", f"{config_dir}/geoextract.txt"]
cfg_geocif = [f"{config_dir}/geobase.txt", f"{config_dir}/countries.txt", f"{config_dir}/crops.txt", f"{config_dir}/geocif.txt"]
```

### geobase.txt

Shared paths, dataset settings, boundary file column mappings, and logging. Key sections:

- **`[DATASETS]`** — Which datasets to download (e.g. `['CHIRPS', 'CPC', 'NDVI', 'ESI', 'NSIDC']`)
- **`[PATHS]`** — All directory paths, derived from `dir_base`
- **Per-dataset sections** (`[CHIRPS]`, `[CPC]`, `[FLDAS]`, etc.) — Dataset-specific settings like data URLs, variables, fill values
- **Boundary file sections** (`[adm_shapefile]`, `[gaul1_asap_v04]`, etc.) — Column mappings from shapefile fields to standard names (`ADM0_NAME`, `ADM1_NAME`, `ADM_ID`)
- **`[DEFAULT]`** — Shared defaults: `start_year`, `end_year`, `parallel_process`, `fraction_cpus`

### countries.txt

Per-country configuration. Each country section specifies boundary file, admin level, seasons, crops, and EO variables. Countries are grouped by calendar category:

- **AMIS countries** — Inherit defaults, override `crops` as needed
- **EWCM countries** — Set `category = EWCM`, `use_cropland_mask = True`, custom `calendar_file` and `boundary_file`
- **`[DEFAULT]`** — Shared defaults including `eo_model` (list of EO variables to extract)

### crops.txt

Crop mask filenames (e.g. `[maize] mask = Percent_Maize.tif`) and calendar category settings (`[EWCM]`, `[AMIS]`).

### geoextract.txt

Extraction settings for geoprepare. `[DEFAULT]` section sets `method`, `redo`, `threshold`, `floor`/`ceil`, `parallel_extract`, `countries`, and `forecast_seasons`.

### geocif.txt

ML and agmet settings for geocif. Contains `[AGMET]` plotting config, per-country crop overrides, ML model definitions, and `[ML]` hyperparameters.

## Supported datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| AEF | AlphaEarth Foundations satellite embeddings (64-band, 10m) | [source.coop](https://source.coop/tge-labs/aef) |
| AGERA5 | Agrometeorological indicators (precipitation, temperature) | [CDS](https://cds.climate.copernicus.eu) |
| AVHRR | Long-term NDVI | NOAA NCEI |
| CHIRPS | Rainfall estimates (v2 and v3) | [CHC](https://www.chc.ucsb.edu/data/chirps) |
| CHIRPS-GEFS | 15-day precipitation forecasts | CHC |
| CPC | Temperature (Tmax, Tmin) and precipitation | NOAA CPC |
| ESI | Evaporative Stress Index (4-week, 12-week) | SERVIR |
| FLDAS | Land surface model outputs (soil moisture, precip, temp) | NASA |
| FPAR | Fraction of Absorbed Photosynthetically Active Radiation | JRC |
| LST | Land Surface Temperature (MODIS MOD11C1) | NASA |
| NDVI | Vegetation index from MODIS (MOD09CMG) | NASA |
| NSIDC | SMAP L4 soil moisture (surface, rootzone) | NASA NSIDC |
| SOIL-MOISTURE | NASA-USDA soil moisture (surface as1, subsurface as2) | NASA |
| VHI | Vegetation Health Index | NOAA STAR |
| VIIRS | Vegetation index from VIIRS (VNP09CMG) | NASA |

### Directory layout

All datasets organize files into year-specific subfolders. After running `geomove` (or on fresh downloads), the directory structure looks like:

```
dir_download/
  nsidc/2025/*.h5, nsidc/2026/*.h5
  chirps_gefs/2026/*.tif
  fpar/2024/*.tif, fpar/2025/*.tif
  modis_lst/*.hdf                     (flat - pymodis manages this)
  ...

dir_intermed/
  cpc_tmax/2024/*.tif, cpc_tmax/2025/*.tif
  cpc_tmin/2024/*.tif, ...
  cpc_precip/2024/*.tif, ...
  chirps/v3/global/2024/*.tif, ...    (CHIRPS already used year subfolders)
  chirps_gefs/2026/*.tif
  esi_4wk/2024/*.tif, ...
  esi_12wk/2024/*.tif, ...
  ndvi/2024/*.tif, ...
  lst/2024/*.tif, ...
  nsidc/subdaily/2025/*.tif
  nsidc/daily/surface/2025/*.tif
  nsidc/daily/rootzone/2025/*.tif
  soil_moisture_as1/2024/*.tif, ...
  soil_moisture_as2/2024/*.tif, ...
  agera5/tif/{variable}/2024/*.tif, ...
  vhi/global/2024/*.tif, ...
  aef/{country}/2018/*.tif, ..., aef/{country}/aef_avg_global.tif
  fldas/.../2024/*.tif, ...           (FLDAS already used year subfolders)
```

## Upload package to PyPI

```bash
# 1. Bump version
uvx bump2version patch --current-version X.X.X --new-version X.X.Y pyproject.toml geoprepare/__init__.py

# 2. Clean, build, upload
rm -rf dist/ build/ *.egg-info/
uv build
uvx twine upload dist/geoprepare-X.X.Y*
```

## Credits

This project was supported by NASA Applied Sciences Grant No. 80NSSC17K0625 through the NASA Harvest Consortium, and the NASA Acres Consortium under NASA Grant #80NSSC23M0034.
