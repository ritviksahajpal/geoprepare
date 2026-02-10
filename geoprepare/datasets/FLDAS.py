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
 3. Reproject/subset to region of interest.

References:
- Arsenault et al. (2020): https://doi.org/10.1175/BAMS-D-18-0264.1
- Shukla et al. (2020): https://doi.org/10.5194/nhess-20-1187-2020
- Hazra et al. (2021): https://doi.org/10.1016/j.jhydrol.2022.129005
"""
import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
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
    dir_interim,
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
        dir_interim: Directory for interim processed data
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
    processed_dir = dir_interim / "fldas" / data_type / "processed" / str(year)
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


def compute_anomalies(
    year,
    month,
    dir_interim,
    variables,
    leads,
    clim_start=1991,
    clim_end=2020,
):
    """
    Compute anomalies relative to climatology.
    
    Args:
        year: Year to compute anomalies for
        month: Month to compute anomalies for
        dir_interim: Directory for interim data
        variables: Variables to process
        leads: Lead times to process
        clim_start: Climatology start year
        clim_end: Climatology end year
    """
    processed_dir = dir_interim / "fldas" / "forecast" / "processed"
    anomaly_dir = dir_interim / "fldas" / "forecast" / "anomaly" / str(year)
    os.makedirs(anomaly_dir, exist_ok=True)
    
    for var in variables:
        for lead in leads:
            # Check if current file exists
            current_file = processed_dir / str(year) / f"fldas_{var}_{year}{month:02d}_lead{lead}.tif"
            if not current_file.exists():
                continue
            
            # Output file
            out_file = anomaly_dir / f"fldas_{var}_{year}{month:02d}_lead{lead}_anomaly.tif"
            if out_file.exists():
                continue
            
            # Collect climatology data for same initialization month and lead
            clim_data = []
            for clim_year in range(clim_start, clim_end + 1):
                clim_file = processed_dir / str(clim_year) / f"fldas_{var}_{clim_year}{month:02d}_lead{lead}.tif"
                
                if clim_file.exists():
                    ds = gdal.Open(str(clim_file))
                    if ds:
                        arr = ds.GetRasterBand(1).ReadAsArray()
                        arr[arr == NODATA_VALUE] = np.nan
                        clim_data.append(arr)
                        ds = None
            
            if len(clim_data) < 10:
                logger.warning(f"Insufficient climatology data for {var} month {month} lead {lead} ({len(clim_data)} years)")
                continue
            
            # Compute climatology mean
            clim_stack = np.stack(clim_data, axis=0)
            clim_mean = np.nanmean(clim_stack, axis=0)
            
            # Read current data
            ds = gdal.Open(str(current_file))
            current_arr = ds.GetRasterBand(1).ReadAsArray()
            current_arr[current_arr == NODATA_VALUE] = np.nan
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            ds = None
            
            # Compute anomaly
            anomaly = current_arr - clim_mean
            anomaly[np.isnan(anomaly)] = NODATA_VALUE
            
            # Save anomaly
            driver = gdal.GetDriverByName("GTiff")
            height, width = anomaly.shape
            dst = driver.Create(
                str(out_file), width, height, 1, gdal.GDT_Float32,
                options=["COMPRESS=LZW", "TILED=YES"]
            )
            dst.SetGeoTransform(geotransform)
            dst.SetProjection(projection)
            band = dst.GetRasterBand(1)
            band.SetNoDataValue(NODATA_VALUE)
            band.WriteArray(anomaly)
            dst.FlushCache()
            dst = None
            
            logger.debug(f"Computed anomaly: {out_file.name}")


# ============================================================================
# Wrapper functions for multiprocessing
# ============================================================================
def _download_fldas_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return download_fldas(*args)


def _extract_and_process_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return extract_and_process(*args)


def _compute_anomalies_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return compute_anomalies(*args)


def run(geoprep):
    """
    Main entry point for FLDAS processing.
    
    Args:
        geoprep: GeoDownload object containing configuration parameters
            Required attributes:
            - start_year: First year to process
            - end_year: Last year to process
            - dir_download: Base download directory
            - dir_interim: Directory for interim processed files
            - redo_last_year: Whether to re-download last year's data
            - parallel_process: Whether to use multiprocessing
            - fraction_cpus: Fraction of CPUs to use for parallel processing
            
            Optional attributes (with defaults):
            - fldas_use_spear: Use NMME with SPEAR model (default: False)
            - fldas_variables: Variables to extract (default: key hydro vars)
            - fldas_leads: Forecast leads to process (default: 0-5)
            - fldas_data_types: Data types to download (default: ['forecast'])
            - fldas_compute_anomalies: Whether to compute anomalies (default: False)
    """
    # Extract parameters from geoprep object
    start_year = geoprep.start_year
    end_year = geoprep.end_year
    dir_download = Path(geoprep.dir_download)
    dir_interim = Path(geoprep.dir_interim)
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
    compute_anom = getattr(geoprep, 'fldas_compute_anomalies', False)
    
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
        (dtype, yr, mon, dir_forecast, dir_openloop, dir_interim, variables, leads, True)
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
    # Step 3: Compute anomalies (optional)
    # =========================================================================
    if compute_anom and "forecast" in data_types:
        anomaly_tasks = [
            (yr, mon, dir_interim, variables, leads, 1991, 2020)
            for yr in range(start_year, end_year + 1)
            for mon in range(1, 13)
        ]
        
        logger.info(f"Computing anomalies for {len(anomaly_tasks)} months...")
        
        if parallel_process:
            with multiprocessing.Pool(num_workers) as p:
                list(
                    tqdm(
                        p.imap_unordered(_compute_anomalies_wrapper, anomaly_tasks),
                        total=len(anomaly_tasks),
                        desc="Compute Anomalies",
                    )
                )
        else:
            for args in tqdm(anomaly_tasks, desc="Compute Anomalies"):
                compute_anomalies(*args)
    
    logger.info("FLDAS processing complete")


if __name__ == "__main__":
    pass