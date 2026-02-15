###############################################################################
# Ritvik Sahajpal
# ritvik@umd.edu
# Updated: Optimized for CDS API v2 with batch downloading
###############################################################################
import os
import itertools
import pathlib
import cdsapi
import urllib3
import pyresample
import rasterio
import multiprocessing
import zipfile
import pandas as pd
import numpy as np
from datetime import timedelta
from calendar import monthrange
from tqdm import tqdm
from affine import Affine
from datetime import datetime
from pathlib import Path

from geoprepare import utils

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

path_template = "template.nc"

profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": None,
    "width": 7200,
    "height": 3600,
    "count": 1,
    "crs": rasterio.crs.CRS({"init": "epsg:4326"}),
    "transform": Affine(0.05, 0.0, -180.0, 0.0, -0.05, 90.0),
    "tiled": False,
    "interleave": "band",
}

variable_names = {
    "Temperature_Air_2m_Mean_24h": ["2m_temperature", "24_hour_mean"],
    "Temperature_Air_2m_Mean_Day_Time": ["2m_temperature", "day_time_mean"],
    "Temperature_Air_2m_Mean_Night_Time": ["2m_temperature", "night_time_mean"],
    "Dew_Point_Temperature_2m_Mean": ["2m_dewpoint_temperature", "24_hour_mean"],
    "Temperature_Air_2m_Max_24h": ["2m_temperature", "24_hour_maximum"],
    "Temperature_Air_2m_Min_24h": ["2m_temperature", "24_hour_minimum"],
    "Temperature_Air_2m_Max_Day_Time": ["2m_temperature", "day_time_maximum"],
    "Temperature_Air_2m_Min_Night_Time": ["2m_temperature", "night_time_minimum"],
    "Precipitation_Flux": ["precipitation_flux", None],
    "Snow_Thickness_Mean": ["snow_thickness", "24_hour_mean"],
    "Solar_Radiation_Flux": ["solar_radiation_flux", None],
    "Vapour_Pressure_Mean": ["vapour_pressure", "24_hour_mean"],
}


def get_list_days(year, month, max_lag_days=10):
    """
    Returns a list of days in a given month, excluding recent days without data.
    
    Args:
        year: the year
        month: the month
        max_lag_days: AgERA5 data lag (default 10 days)
        
    Returns:
        List of day strings formatted as '01', '02', etc.
    """
    num_days = monthrange(year, month)[1]
    cutoff_date = datetime.now() - timedelta(days=max_lag_days)
    
    days = []
    for day in range(1, num_days + 1):
        current_date = datetime(year, month, day)
        if current_date <= cutoff_date:
            days.append(str(day).zfill(2))
    
    return days


def create_target_fname(meteo_variable_full_name, sday, agera5_dir, stat="final", v="2.0"):
    """
    Creates the AgERA5 variable filename for given variable and day.
    
    Args:
        meteo_variable_full_name: the full name of the meteo variable
        sday: the date string (YYYYMMDD format)
        agera5_dir: the path to the AgERA5 dataset
        stat: statistic type (default 'final')
        v: version string (default '2.0')
        
    Returns:
        Path to the target filename
    """
    name_with_dashes = meteo_variable_full_name.replace("_", "-")
    nc_fname = Path(agera5_dir) / f"{name_with_dashes}_C3S-glob-agric_AgERA5_{sday}_{stat}-v{v}.nc"
    return nc_fname


def get_date_from_fname(path_file):
    """
    Extract date from AgERA5 filename.
    
    Args:
        path_file: path to the file
        
    Returns:
        datetime object
    """
    fname = os.path.basename(path_file)
    date_string = fname.split("_")[-2]
    date = datetime.strptime(date_string, "%Y%m%d")
    return date


def get_missing_days(year, month, varname, path_nc, max_lag_days=10):
    """
    Get list of days that haven't been downloaded yet.
    
    Args:
        year: the year
        month: the month
        varname: variable name
        path_nc: path to NC files directory
        max_lag_days: data lag in days
        
    Returns:
        List of missing day strings
    """
    all_days = get_list_days(year, month, max_lag_days)
    missing_days = []
    
    for day in all_days:
        sday = f"{year}{str(month).zfill(2)}{day}"
        fname = create_target_fname(varname, sday, path_nc / varname, v="2.0")
        if not os.path.exists(fname):
            missing_days.append(day)
    
    return missing_days


def remap_like(original_nc, target_nc, name_var, index=0):
    """
    Remap data from original grid to target grid.
    
    Args:
        original_nc: source NetCDF file/handle
        target_nc: target NetCDF file/handle
        name_var: variable name to remap
        index: time index (default 0)
        
    Returns:
        Remapped array
    """
    hndl_original = utils.convert_to_nc_hndl(original_nc)
    hndl_target = utils.convert_to_nc_hndl(target_nc)

    lat = hndl_original.variables["lat"].values
    lon = hndl_original.variables["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    orig_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    lat = hndl_target.variables["latitude"].values
    lon = hndl_target.variables["longitude"].values
    lon2d, lat2d = np.meshgrid(lon - 180, lat)
    targ_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    var = hndl_original.variables[name_var].values[index]
    remapped_var = pyresample.kd_tree.resample_nearest(
        orig_def, var, targ_def, radius_of_influence=5000, fill_value=None
    )

    return remapped_var


def process_agERA5(all_params):
    """
    Process a single AgERA5 NetCDF file to GeoTIFF.
    
    Args:
        all_params: tuple of (variable_name, nc_input_path, output_directory)
    """
    var, nc_input, dir_output = all_params

    date = get_date_from_fname(nc_input)
    year = date.year

    fl_out = f"agera5_{year}{str(pd.DatetimeIndex([date]).dayofyear[0]).zfill(3)}_{var}_global.tif"

    if not os.path.isfile(dir_output / fl_out):
        arr = remap_like(
            nc_input,
            pathlib.Path(__file__).parent.resolve() / path_template,
            name_var=var,
        )
        utils.arr_to_tif(arr, dir_output / fl_out, profile)


def parallel_process_agERA5(params):
    """
    Process all downloaded AgERA5 NetCDF files to GeoTIFF in parallel.
    
    Args:
        params: configuration parameters object
    """
    all_params = []

    for var in list(variable_names.keys()):
        if var not in params.variables:
            continue
            
        dir_output = params.dir_intermed / "agera5" / "tif" / var
        os.makedirs(dir_output, exist_ok=True)

        dir_nc = params.dir_intermed / "agera5" / "nc" / var

        nc_files = list(dir_nc.glob("*.nc"))
        for nc_input in nc_files:
            all_params.append((var, nc_input, dir_output))

    if not all_params:
        params.logger.info("No NetCDF files to process")
        return

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            list(tqdm(
                p.imap_unordered(process_agERA5, all_params),
                total=len(all_params),
                desc="Processing AgERA5 to TIF"
            ))
    else:
        for val in tqdm(all_params, desc="Processing AgERA5 to TIF"):
            process_agERA5(val)


def download_nc_batch(inputs):
    """
    Download AgERA5 data for an entire month in a single API call.
    
    Args:
        inputs: tuple of (cdsapi_client, params, path_download, path_nc, varname, year, month)
    """
    c, params, path_download, path_nc, varname, year, mon = inputs
    
    os.makedirs(path_download / varname, exist_ok=True)
    os.makedirs(path_nc / varname, exist_ok=True)

    # Get only missing days to avoid re-downloading
    missing_days = get_missing_days(year, mon, varname, path_nc)
    
    if not missing_days:
        return  # All days already downloaded
    
    statistic = variable_names.get(varname)[1]
    variable = variable_names.get(varname)[0]
    
    # Create zip filename for this batch
    fzip = path_download / varname / f"{varname}_{year}{str(mon).zfill(2)}_batch.zip"
    
    try:
        # Build request with all missing days at once
        request = {
            "version": "2_0",
            "variable": variable,
            "year": str(year),
            "month": str(mon).zfill(2),
            "day": missing_days,
        }
        
        # Add statistic if applicable (some variables don't have it)
        if statistic:
            request["statistic"] = statistic
        
        params.logger.info(f"Downloading {varname} {year}-{mon:02d}: {len(missing_days)} days")
        
        # Use new CDS API download pattern
        c.retrieve("sis-agrometeorological-indicators", request).download(str(fzip))
        
        # Extract all NC files from zip
        if os.path.isfile(fzip):
            with zipfile.ZipFile(fzip, "r") as zip_ref:
                zip_ref.extractall(path_nc / varname)
            
            # Optionally remove zip file to save space
            # os.remove(fzip)
                
    except Exception as e:
        params.logger.error(f"Could not download {varname} {year}-{mon:02d}: {e}")


def download_parallel_nc(c, params, path_download, path_nc, variable):
    """
    Download AgERA5 data in parallel by month.
    
    Args:
        c: cdsapi Client
        params: configuration parameters
        path_download: download directory
        path_nc: NetCDF output directory
        variable: variable name to download
    """
    all_params = []
    
    for year in range(params.start_year, params.end_year + 1):
        for mon in range(1, 13):
            # Skip future months
            if datetime(year, mon, 1) > datetime.now():
                continue
            
            # Check if any days are missing before adding to queue
            missing_days = get_missing_days(year, mon, variable, path_nc)
            if missing_days:
                all_params.append((c, params, path_download, path_nc, variable, year, mon))

    if not all_params:
        params.logger.info(f"All {variable} data already downloaded")
        return

    params.logger.info(f"Downloading {variable}: {len(all_params)} months with missing data")

    # Note: CDS API has rate limits, so parallel downloads may not always be faster
    # Consider using sequential for more reliable downloads
    if params.parallel_process and len(all_params) > 1:
        # Use fewer workers for API calls to avoid rate limiting
        num_workers = min(int(multiprocessing.cpu_count() * params.fraction_cpus), 4)
        with multiprocessing.Pool(num_workers) as p:
            list(tqdm(
                p.imap_unordered(download_nc_batch, all_params),
                total=len(all_params),
                desc=f"Downloading AgERA5 {variable}"
            ))
    else:
        for param in tqdm(all_params, desc=f"Downloading AgERA5 {variable}"):
            download_nc_batch(param)


def run(params):
    """
    Main entry point for AgERA5 data download and processing.
    
    Args:
        params: configuration object with attributes:
            - dir_download: download directory
            - dir_intermed: interim data directory
            - start_year: first year to download
            - end_year: last year to download
            - variables: list of variables to download
            - parallel_process: whether to use parallel processing
            - fraction_cpus: fraction of CPUs to use
            - logger: logging object
    """
    path_download = params.dir_download / "agera5"
    path_nc = params.dir_intermed / "agera5" / "nc"

    try:
        c = cdsapi.Client()
    except Exception as e:
        params.logger.error(f"Cannot connect to CDSAPI: {e}")
        params.logger.error("Make sure ~/.cdsapirc is configured with your CDS API key")
        return

    # Download requested variables
    for variable in variable_names.keys():
        if variable in params.variables:
            download_parallel_nc(c, params, path_download, path_nc, variable)

    # Process NC to TIF
    parallel_process_agERA5(params)


if __name__ == "__main__":
    pass