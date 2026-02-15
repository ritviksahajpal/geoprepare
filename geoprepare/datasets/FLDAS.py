#!/usr/bin/env python3
"""
Ritvik Sahajpal
ritvik@umd.edu

Download and process FLDAS-NMME hydrological forecast data from NASA NCCS.

Data source: https://portal.nccs.nasa.gov/datashare/fldas-forecast/Data_files/FF2/
Documentation: https://ldas.gsfc.nasa.gov/fldas/models/forecast

Directory structure on server:
  NMME_noSPEAR/FCST/{YYYYMM}/NMME_hydro_{YYYYMM}.nc
  OL/{YYYYMM}/OL_hydro_{YYYYMM}.nc

The FLDAS forecast system uses:
- NoahMP4.0.1 LSM forced by NMME precipitation ensemble + CFSv2 non-precip forcings
- NMME models: CanESM5, GEM5.2-NEMO, CESM1, CFSv2, GEOSv2
- 6-month lead forecasts (lead 0-5) at 0.25° resolution
- Coverage: Continental Africa and Middle East (320x320 grid)

Data products:
- FCST (forecast): NMME ensemble averaged monthly hydrologic forecasts
  Files: NMME_hydro_YYYYMM.nc (6 lead months per file)
- OL (openloop): Monthly averaged hydrologic output forced by CHIRPS + MERRA-2
  Files: OL_hydro_YYYYMM.nc

Variables:
- SoilMoist_tavg: Soil moisture content (3 profiles) [m³/m³]
- Evap_tavg: Total evapotranspiration [kg m⁻² s⁻¹]
- Qs_tavg: Surface runoff [kg m⁻² s⁻¹]
- Qsb_tavg: Subsurface runoff [kg m⁻² s⁻¹]
- TWS_tavg: Terrestrial water storage [mm]
- Streamflow_tavg: Streamflow [m³/s]
- Rainf_tavg: Rainfall flux [kg m⁻² s⁻¹]
- TotalPrecip_tavg: Total precipitation [kg m⁻² s⁻¹]
- Tair_tavg: Air temperature [K]
- SWdown_tavg: Surface downward shortwave radiation [W/m²]
- Qle_tavg: Latent heat flux [W/m²]
- Qh_tavg: Sensible heat flux [W/m²]

Steps:
 1. Download NetCDF files for each year/month.
 2. Extract variables and convert units.
 3. Reproject to 0.05° global grid (3600x7200) matching CHIRPS, CPC, ESI, etc.
"""
import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pyresample
import xarray as xr
import requests
from osgeo import gdal, osr
from tqdm import tqdm

# Module-level logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# Data product types
LIST_PRODUCTS = ["forecast", "openloop"]

# NoData value for output files
NODATA_VALUE = -9999

# Base URL for FLDAS data
BASE_URL = "https://portal.nccs.nasa.gov/datashare/fldas-forecast/Data_files/FF2/"

# Available variables in FLDAS forecast files
FLDAS_VARIABLES = [
    "SoilMoist_tavg",   # Soil moisture (3 layers)
    "Evap_tavg",        # Evapotranspiration
    "Qs_tavg",          # Surface runoff
    "Qsb_tavg",         # Subsurface runoff
    "TWS_tavg",         # Terrestrial water storage
    "Streamflow_tavg",  # Streamflow
    "Rainf_tavg",       # Rainfall
    "TotalPrecip_tavg", # Total precipitation
    "Tair_tavg",        # Air temperature
    "SWdown_tavg",      # Shortwave radiation
    "Qle_tavg",         # Latent heat flux
    "Qh_tavg",          # Sensible heat flux
]

# Unit conversion factors (from native to common units)
UNIT_CONVERSIONS = {
    # Flux rates to mm/day
    "Evap_tavg": {"factor": 86400, "target_unit": "mm/day"},
    "Qs_tavg": {"factor": 86400, "target_unit": "mm/day"},
    "Qsb_tavg": {"factor": 86400, "target_unit": "mm/day"},
    "Rainf_tavg": {"factor": 86400, "target_unit": "mm/day"},
    "TotalPrecip_tavg": {"factor": 86400, "target_unit": "mm/day"},
    # Temperature K to C
    "Tair_tavg": {"factor": 1, "offset": -273.15, "target_unit": "C"},
}


def get_fldas_url(data_type, year, month, use_spear=False):
    """
    Get the URL for a specific FLDAS file.
    
    Server structure:
      NMME_noSPEAR/FCST/{YYYYMM}/NMME_hydro_{YYYYMM}.nc
      NMME/FCST/{YYYYMM}/NMME_hydro_{YYYYMM}.nc  (with SPEAR)
      OL/{YYYYMM}/OL_hydro_{YYYYMM}.nc
    
    Args:
        data_type: 'forecast' or 'openloop'
        year: Year (int)
        month: Month (int)
        use_spear: Whether to use NMME version with SPEAR model (default: False)
    
    Returns:
        URL string for the specific file
    """
    yyyymm = f"{year:04d}{month:02d}"
    
    if data_type == "forecast":
        subdir = "NMME" if use_spear else "NMME_noSPEAR"
        return f"{BASE_URL}{subdir}/FCST/{yyyymm}/NMME_hydro_{yyyymm}.nc"
    else:  # openloop
        return f"{BASE_URL}OL/{yyyymm}/OL_hydro_{yyyymm}.nc"


def get_fldas_filename(data_type, year, month):
    """
    Get the filename for a FLDAS file.
    
    Args:
        data_type: 'forecast' or 'openloop'
        year: Year
        month: Month
    
    Returns:
        Filename string
    """
    yyyymm = f"{year:04d}{month:02d}"
    if data_type == "forecast":
        return f"NMME_hydro_{yyyymm}.nc"
    else:  # openloop
        return f"OL_hydro_{yyyymm}.nc"


def download_fldas(
    data_type, year, month, dir_forecast, dir_openloop, redo_last_year, use_spear
):
    """
    Download a single FLDAS NetCDF file for a given year/month.
    
    Args:
        data_type: 'forecast' or 'openloop'
        year: Year to download
        month: Month to download
        dir_forecast: Directory for forecast data
        dir_openloop: Directory for openloop data
        redo_last_year: Whether to re-download last year's data
        use_spear: Whether to use NMME version with SPEAR model
    """
    # Skip old data if not re-downloading (keep last 2 years)
    current_year = datetime.today().year
    if not redo_last_year and year < current_year - 1:
        # Still download if file doesn't exist
        pass

    # Set output directory with year subfolder
    if data_type == "forecast":
        out_dir = dir_forecast / str(year)
    else:
        out_dir = dir_openloop / str(year)

    os.makedirs(out_dir, exist_ok=True)
    
    # Get URL and filename
    url = get_fldas_url(data_type, year, month, use_spear)
    filename = get_fldas_filename(data_type, year, month)
    out_file = out_dir / filename
    
    # Skip if already exists
    if out_file.exists():
        logger.debug(f"File exists, skipping: {out_file.name}")
        return
    
    # Download file
    try:
        logger.info(f"Downloading {url}")
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        with open(out_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded: {out_file.name}")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.debug(f"File not found (404): {url}")
        else:
            logger.error(f"HTTP error downloading {url}: {e}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")


def extract_and_process(
    data_type,
    year,
    month,
    dir_forecast,
    dir_openloop,
    dir_intermed,
    variables,
    leads,
    convert_units=True,
):
    """
    Extract variables from FLDAS NetCDF and process to GeoTIFF.
    
    Args:
        data_type: 'forecast' or 'openloop'
        year: Year
        month: Month
        dir_forecast: Directory for forecast data
        dir_openloop: Directory for openloop data
        dir_intermed: Directory for interim processed data
        variables: List of variables to extract
        leads: List of lead times to extract (forecast only)
        convert_units: Whether to convert units
    """
    # Set source directory
    if data_type == "forecast":
        src_dir = Path(dir_forecast) / str(year)
    else:
        src_dir = Path(dir_openloop) / str(year)
    
    src_filename = get_fldas_filename(data_type, year, month)
    src_path = src_dir / src_filename
    
    if not src_path.exists():
        return
    
    # Set up output directories
    processed_dir = dir_intermed / "fldas" / data_type / "processed" / str(year)
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        ds = xr.open_dataset(src_path)
    except Exception as e:
        logger.error(f"Failed to open {src_path}: {e}")
        return
    
    # Process each variable
    for var in variables:
        if var not in ds.data_vars:
            continue
        
        data = ds[var]
        
        # Handle soil moisture with multiple layers
        if "SoilProfile" in data.dims:
            # Extract surface layer only (layer 0)
            data = data.isel(SoilProfile=0)
        
        # Handle forecast leads
        if "Lead" in data.dims and data_type == "forecast":
            for lead in leads:
                if lead >= len(ds.Lead):
                    continue
                
                # Output filename
                out_tif = processed_dir / f"fldas_{var}_{year}{month:02d}_lead{lead}.tif"
                
                # Skip if already processed
                if out_tif.exists():
                    continue
                    
                lead_data = data.isel(Lead=lead, Time=0)
                
                # Apply unit conversion
                if convert_units and var in UNIT_CONVERSIONS:
                    conv = UNIT_CONVERSIONS[var]
                    lead_data = lead_data * conv["factor"]
                    if "offset" in conv:
                        lead_data = lead_data + conv["offset"]
                
                # Convert to numpy and handle nodata
                arr = lead_data.values.astype(np.float32)
                arr[np.isnan(arr)] = NODATA_VALUE
                arr[arr == -9999.0] = NODATA_VALUE
                
                # Save as GeoTIFF
                _write_geotiff(
                    arr,
                    out_tif,
                    ds.Lat.values,
                    ds.Lon.values,
                )
                logger.debug(f"Processed {out_tif.name}")
        else:
            # Openloop data (no leads)
            out_tif = processed_dir / f"fldas_{var}_{year}{month:02d}.tif"
            
            # Skip if already processed
            if out_tif.exists():
                continue
            
            if "Time" in data.dims:
                data = data.isel(Time=0)
            
            # Apply unit conversion
            if convert_units and var in UNIT_CONVERSIONS:
                conv = UNIT_CONVERSIONS[var]
                data = data * conv["factor"]
                if "offset" in conv:
                    data = data + conv["offset"]
            
            arr = data.values.astype(np.float32)
            arr[np.isnan(arr)] = NODATA_VALUE
            arr[arr == -9999.0] = NODATA_VALUE
            
            _write_geotiff(
                arr,
                out_tif,
                ds.Lat.values,
                ds.Lon.values,
            )
            logger.debug(f"Processed {out_tif.name}")
    
    ds.close()


def _write_geotiff(arr, out_path, lats, lons):
    """
    Write array to GeoTIFF with proper georeferencing.
    
    Args:
        arr: 2D numpy array
        out_path: Output file path
        lats: Latitude coordinate array
        lons: Longitude coordinate array
    """
    height, width = arr.shape
    
    # Calculate geotransform
    # FLDAS grid is 0.25° resolution
    lat_res = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.25
    lon_res = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.25
    
    # Determine if latitude is ascending or descending
    lat_ascending = lats[0] < lats[-1] if len(lats) > 1 else False
    
    if lat_ascending:
        # Flip array for north-up orientation
        arr = np.flipud(arr)
        ul_lat = lats[-1] + lat_res / 2
    else:
        ul_lat = lats[0] + lat_res / 2
    
    ul_lon = lons[0] - lon_res / 2
    
    # Create GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(out_path), width, height, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"]
    )
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst.SetProjection(srs.ExportToWkt())
    dst.SetGeoTransform((ul_lon, lon_res, 0, ul_lat, 0, -lat_res))
    
    band = dst.GetRasterBand(1)
    band.SetNoDataValue(NODATA_VALUE)
    band.WriteArray(arr)
    
    dst.FlushCache()
    dst = None


# Standard 0.05° global grid (3600x7200) matching other EO datasets
GLOBAL_HEIGHT = 3600
GLOBAL_WIDTH = 7200
GLOBAL_RES = 0.05
GLOBAL_GEOTRANSFORM = (-180, GLOBAL_RES, 0, 90, 0, -GLOBAL_RES)


def reproject_to_global(
    data_type,
    year,
    month,
    dir_intermed,
    variables,
    leads,
):
    """
    Reproject processed FLDAS GeoTIFFs from native 0.25° to 0.05° global grid.

    Uses nearest-neighbor resampling via pyresample to match the standard
    3600x7200 global grid used by CHIRPS, CPC, ESI, etc.

    Args:
        data_type: 'forecast' or 'openloop'
        year: Year
        month: Month
        dir_intermed: Directory for interim processed data
        variables: List of variables to reproject
        leads: List of lead times (forecast only)
    """
    processed_dir = Path(dir_intermed) / "fldas" / data_type / "processed" / str(year)
    global_dir = Path(dir_intermed) / "fldas" / "global" / str(year)
    os.makedirs(global_dir, exist_ok=True)

    # Build list of (input, output) file pairs
    file_pairs = []
    for var in variables:
        if data_type == "forecast":
            for lead in leads:
                src = processed_dir / f"fldas_{var}_{year}{month:02d}_lead{lead}.tif"
                dst = global_dir / f"fldas_{var}_{year}{month:02d}_lead{lead}_global.tif"
                file_pairs.append((src, dst))
        else:
            src = processed_dir / f"fldas_{var}_{year}{month:02d}.tif"
            dst = global_dir / f"fldas_{var}_{year}{month:02d}_global.tif"
            file_pairs.append((src, dst))

    for src_path, dst_path in file_pairs:
        if dst_path.exists() or not src_path.exists():
            continue

        try:
            ds = gdal.Open(str(src_path))
            if ds is None:
                logger.error(f"Failed to open {src_path}")
                continue

            band = ds.GetRasterBand(1)
            src_arr = band.ReadAsArray()
            gt = ds.GetGeoTransform()
            src_height, src_width = src_arr.shape

            # Build source lat/lon arrays from geotransform
            src_lons = np.array([gt[0] + (j + 0.5) * gt[1] for j in range(src_width)])
            src_lats = np.array([gt[3] + (i + 0.5) * gt[5] for i in range(src_height)])
            src_lon2d, src_lat2d = np.meshgrid(src_lons, src_lats)

            ds = None

            # Build target 0.05° global grid
            tgt_lons = np.linspace(-180 + GLOBAL_RES / 2, 180 - GLOBAL_RES / 2, GLOBAL_WIDTH)
            tgt_lats = np.linspace(90 - GLOBAL_RES / 2, -90 + GLOBAL_RES / 2, GLOBAL_HEIGHT)
            tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lons, tgt_lats)

            # Define swath geometries
            src_def = pyresample.geometry.SwathDefinition(
                lons=src_lon2d, lats=src_lat2d
            )
            tgt_def = pyresample.geometry.SwathDefinition(
                lons=tgt_lon2d, lats=tgt_lat2d
            )

            # Mask nodata before resampling
            src_masked = np.where(src_arr == NODATA_VALUE, np.nan, src_arr)

            # Nearest-neighbor resample to global grid
            out_arr = pyresample.kd_tree.resample_nearest(
                src_def,
                src_masked,
                tgt_def,
                radius_of_influence=30000,  # 30 km (~0.25° at equator)
                fill_value=np.nan,
            )

            # Restore nodata
            out_arr = np.where(np.isnan(out_arr), NODATA_VALUE, out_arr).astype(np.float32)

            # Write global GeoTIFF
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(
                str(dst_path), GLOBAL_WIDTH, GLOBAL_HEIGHT, 1, gdal.GDT_Float32,
                options=["COMPRESS=LZW", "TILED=YES"]
            )
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            dst_ds.SetProjection(srs.ExportToWkt())
            dst_ds.SetGeoTransform(GLOBAL_GEOTRANSFORM)
            dst_ds.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
            dst_ds.GetRasterBand(1).WriteArray(out_arr)
            dst_ds.FlushCache()
            dst_ds = None

            logger.debug(f"Reprojected to global: {dst_path.name}")

        except Exception as e:
            logger.error(f"Failed to reproject {src_path}: {e}")


# ============================================================================
# Wrapper functions for multiprocessing
# ============================================================================
def _download_fldas_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return download_fldas(*args)


def _extract_and_process_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return extract_and_process(*args)


def _reproject_to_global_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return reproject_to_global(*args)


def run(geoprep):
    """
    Main entry point for FLDAS processing.
    
    Args:
        geoprep: GeoDownload object containing configuration parameters
            Required attributes:
            - start_year: First year to process
            - end_year: Last year to process
            - dir_download: Base download directory
            - dir_intermed: Directory for interim processed files
            - redo_last_year: Whether to re-download last year's data
            - parallel_process: Whether to use multiprocessing
            - fraction_cpus: Fraction of CPUs to use for parallel processing
            
            Optional attributes (with defaults):
            - fldas_use_spear: Use NMME with SPEAR model (default: False)
            - fldas_variables: Variables to extract (default: key hydro vars)
            - fldas_leads: Forecast leads to process (default: 0-5)
            - fldas_data_types: Data types to download (default: ['forecast'])
    """
    # Extract parameters from geoprep object
    start_year = geoprep.start_year
    end_year = geoprep.end_year
    dir_download = Path(geoprep.dir_download)
    dir_intermed = Path(geoprep.dir_intermed)
    redo_last_year = geoprep.redo_last_year
    parallel_process = geoprep.parallel_process
    fraction_cpus = geoprep.fraction_cpus
    
    # Get optional FLDAS-specific parameters with defaults
    use_spear = getattr(geoprep, 'fldas_use_spear', False)
    variables = getattr(geoprep, 'fldas_variables', [
        "SoilMoist_tavg",
        "TotalPrecip_tavg", 
        "Tair_tavg",
        "Evap_tavg",
        "TWS_tavg",
    ])
    leads = getattr(geoprep, 'fldas_leads', list(range(6)))
    data_types = getattr(geoprep, 'fldas_data_types', ['forecast'])
    
    logger.info(f"Processing FLDAS data: {data_types}")
    logger.info(f"Variables: {variables}")
    logger.info(f"Leads: {leads}")
    logger.info(f"Use SPEAR model: {use_spear}")
    logger.info(f"Years: {start_year} - {end_year}")

    # Set up directories
    dir_forecast = dir_download / 'fldas' / 'forecast'
    dir_openloop = dir_download / 'fldas' / 'openloop'
    dir_forecast.mkdir(parents=True, exist_ok=True)
    dir_openloop.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, int(multiprocessing.cpu_count() * fraction_cpus))

    # =========================================================================
    # Step 1: Download data (one file per year/month)
    # =========================================================================
    download_tasks = [
        (dtype, yr, mon, dir_forecast, dir_openloop, redo_last_year, use_spear)
        for dtype in data_types
        for yr in range(start_year, end_year + 1)
        for mon in range(1, 13)
    ]
    
    logger.info(f"Downloading {len(download_tasks)} files...")
    
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(
                tqdm(
                    p.imap_unordered(_download_fldas_wrapper, download_tasks),
                    total=len(download_tasks),
                    desc="Download FLDAS",
                )
            )
    else:
        for args in tqdm(download_tasks, desc="Download FLDAS"):
            download_fldas(*args)

    # =========================================================================
    # Step 2: Extract and process variables to GeoTIFF
    # =========================================================================
    process_tasks = [
        (dtype, yr, mon, dir_forecast, dir_openloop, dir_intermed, variables, leads, True)
        for dtype in data_types
        for yr in range(start_year, end_year + 1)
        for mon in range(1, 13)
    ]
    
    logger.info(f"Processing {len(process_tasks)} files...")
    
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(
                tqdm(
                    p.imap_unordered(_extract_and_process_wrapper, process_tasks),
                    total=len(process_tasks),
                    desc="Process FLDAS",
                )
            )
    else:
        for args in tqdm(process_tasks, desc="Process FLDAS"):
            extract_and_process(*args)

    # =========================================================================
    # Step 3: Reproject to 0.05° global grid (3600x7200)
    # Matches standard grid used by CHIRPS, CPC, ESI, etc.
    # =========================================================================
    reproj_tasks = [
        (dtype, yr, mon, dir_intermed, variables, leads)
        for dtype in data_types
        for yr in range(start_year, end_year + 1)
        for mon in range(1, 13)
    ]

    logger.info(f"Reprojecting {len(reproj_tasks)} files to global grid...")

    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(
                tqdm(
                    p.imap_unordered(_reproject_to_global_wrapper, reproj_tasks),
                    total=len(reproj_tasks),
                    desc="Reproject FLDAS",
                )
            )
    else:
        for args in tqdm(reproj_tasks, desc="Reproject FLDAS"):
            reproject_to_global(*args)
    
    logger.info("FLDAS processing complete")


if __name__ == "__main__":
    pass