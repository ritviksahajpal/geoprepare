###############################################################################
# NSIDC SMAP L4 Soil Moisture Data Download and Processing
# Ritvik Sahajpal, ritvik@umd.edu
#
# Downloads SMAP L4 Global 3-hourly 9 km Surface and Root Zone Soil Moisture
# from NASA NSIDC DAAC (version 008)
#
# Requires NASA Earthdata credentials stored in .netrc file:
#    machine urs.earthdata.nasa.gov login MYUSERNAME password MYPASSWORD
###############################################################################
import os
import netrc
import time
import requests
import arrow as ar
import pandas as pd
import numpy as np
import rasterio as rio
from pathlib import Path
from tqdm import tqdm
from osgeo import gdal
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse
from pyl4c.spatial import array_to_raster
from pyl4c.data.fixtures import EASE2_GRID_PARAMS
from pyl4c.epsg import EPSG

# SMAP L4 product configuration
SHORT_NAME = "SPL4SMGP"
VERSION = "008"
BASE_URL = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/SMAP/SPL4SMGP/008"
URS_URL = "https://urs.earthdata.nasa.gov"

# 8 files per day (every 3 hours: 01:30, 04:30, 07:30, 10:30, 13:30, 16:30, 19:30, 22:30)
TIME_STAMPS = ["013000", "043000", "073000", "103000", "133000", "163000", "193000", "223000"]

VAR_LIST = ["sm_surface", "sm_rootzone"]


def get_credentials():
    """
    Get NASA Earthdata credentials from .netrc file.
    
    Returns:
        tuple: (username, password) or (None, None) if not found
    """
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
        return username, password
    except Exception:
        return None, None


def get_file_urls_for_day(date):
    """
    Generate file URLs for a specific day (8 files per day, 3-hourly).
    
    Args:
        date: datetime object for the day
        
    Returns:
        List of (url, filename) tuples
    """
    urls = []
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    date_str = date.strftime("%Y%m%d")

    for time_stamp in TIME_STAMPS:
        filename = f"SMAP_L4_SM_gph_{date_str}T{time_stamp}_Vv8011_001.h5"
        url = f"{BASE_URL}/{year}/{month}/{day}/{filename}"
        urls.append((url, filename))

    return urls


def download_file(session, url, filename, dir_out, logger, max_retries=3):
    """
    Download a single file with retry logic.
    
    Args:
        session: requests Session with authentication
        url: URL of the file to download
        filename: Name of the file
        dir_out: Output directory
        logger: Logger instance
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    filepath = dir_out / filename

    # Skip if file already exists
    if os.path.exists(filepath):
        return True

    for attempt in range(max_retries):
        try:
            response = session.get(url, stream=True, timeout=60)

            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True

            elif response.status_code == 401:
                logger.error(f"Authentication failed for {filename}. Check credentials in .netrc")
                return False

            elif response.status_code == 404:
                # File not yet available - this is normal for recent dates
                return False

            else:
                logger.warning(f"Error {response.status_code} downloading {filename}: {response.reason}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Download error for {filename} (attempt {attempt + 1}/{max_retries}): {str(e)}")

        # Wait before retry
        if attempt < max_retries - 1:
            time.sleep(5)

    return False


def download_date_range(params, start_date, end_date):
    """
    Download SMAP data for a date range.
    
    Args:
        params: Configuration parameters object
        start_date: Start date (datetime or arrow object)
        end_date: End date (datetime or arrow object)
    """
    dir_out = params.dir_download / "nsidc"
    os.makedirs(dir_out, exist_ok=True)

    # Get credentials
    username, password = get_credentials()
    if not username or not password:
        params.logger.error("NASA Earthdata credentials not found in .netrc file")
        params.logger.error("Add to ~/.netrc: machine urs.earthdata.nasa.gov login USERNAME password PASSWORD")
        return

    # Create authenticated session
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)

    # Convert to datetime if needed
    if hasattr(start_date, "datetime"):
        start_date = start_date.datetime
    if hasattr(end_date, "datetime"):
        end_date = end_date.datetime

    # Iterate through date range
    current_date = start_date
    total_downloaded = 0
    total_failed = 0

    while current_date <= end_date:
        urls = get_file_urls_for_day(current_date)

        for url, filename in tqdm(urls, desc=f"Downloading {current_date.strftime('%Y-%m-%d')}", leave=False):
            if download_file(session, url, filename, dir_out, params.logger):
                total_downloaded += 1
            else:
                total_failed += 1

            # Small delay between downloads
            time.sleep(0.3)

        current_date += timedelta(days=1)

    params.logger.info(f"Download complete: {total_downloaded} files downloaded, {total_failed} failed/unavailable")


def download(params):
    """
    Download SMAP data based on params configuration.
    
    Args:
        params: Configuration parameters object with:
            - start_year, end_year: Year range
            - dir_download: Download directory
            - logger: Logger instance
    """
    # Determine date range
    start_date = datetime(params.start_year, 1, 1)
    
    # End date is either end of end_year or today, whichever is earlier
    end_of_year = datetime(params.end_year, 12, 31)
    today = datetime.now()
    end_date = min(end_of_year, today)

    params.logger.info(f"Downloading SMAP L4 data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    download_date_range(params, start_date, end_date)


def convert(params, source_h5):
    """
    Converts a SMAP L4 HDF file to GeoTIFF format.
    
    Args:
        params: Configuration parameters object
        source_h5: Path to source H5 file
    """
    dir_subdaily = params.dir_intermed / "nsidc" / "subdaily"
    os.makedirs(dir_subdaily, exist_ok=True)

    inp_file_name = os.path.basename(source_h5)
    
    # Parse filename: SMAP_L4_SM_gph_20250101T013000_Vv8011_001.h5
    # Positions: year=15:19, month=19:21, day=21:23, time=24:30
    year = inp_file_name[15:19]
    month = inp_file_name[19:21]
    day = inp_file_name[21:23]
    hhmmss = inp_file_name[24:30]

    # Get day of year from year, month, day
    doy = ar.get(f"{year}-{month}-{day}").format("DDD")

    # Options for gdal_translate
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        outputSRS="+proj=cea +lon_0=0 +lat_ts=30 +ellps=WGS84 +units=m",
        outputBounds=[-17367530.45, 7314540.11, 17367530.45, -7314540.11],
    )

    # array_to_raster params
    gt = EASE2_GRID_PARAMS["M09"]["geotransform"]
    wkt = EPSG[6933]

    # Temp tiff list for cleanup
    tif_list = []

    # Convert individual variables to separate GeoTiff files
    for iband, var in enumerate(VAR_LIST):
        # Output filename format
        ras_subdaily = dir_subdaily / f"nasa_usda_soil_moisture_{year}{doy.zfill(3)}T{hhmmss}_{var}_global.tif"

        if os.path.isfile(ras_subdaily):
            continue

        sds = gdal.Open(f"HDF5:{str(source_h5)}://Geophysical_Data/{var}")
        if sds is None:
            continue
            
        try:
            sds_array = sds.ReadAsArray()
        except Exception as e:
            params.logger.warning(f"Could not read {var} from {source_h5}: {e}")
            continue

        dst_tmp = str(params.dir_intermed) + os.sep + f"{iband+1}_{year}{doy.zfill(3)}T{hhmmss}_{var}.tif"
        sds_gdal = array_to_raster(sds_array, gt, wkt)

        ds = gdal.Translate(dst_tmp, sds_gdal, options=translate_options)
        ds = None
        tif_list.append(dst_tmp)

        # Warp to WGS84
        ds = gdal.Warp(
            str(ras_subdaily),
            dst_tmp,
            options="-overwrite -co compress=LZW -srcnodata -999.0 -dstnodata 9999.0 -of Gtiff -r bilinear -s_srs EPSG:6933 -t_srs EPSG:4326 -te -180 -90 180 90 -ts 7200 3600",
        )
        ds = None

    # Remove temporary files
    for f in tif_list:
        if os.path.exists(f):
            os.remove(f)


def read_file(file):
    """Read a single band from a raster file."""
    with rio.open(file) as src:
        return src.read(1)


def create_daily_sm_file(rasters, output_file):
    """
    Create daily average soil moisture file from subdaily files.
    
    Args:
        rasters: List of subdaily raster file paths
        output_file: Output file path
    """
    # Read all data as a list of numpy arrays
    array_list = [read_file(x) for x in rasters]
    # Perform averaging
    array_out = np.mean(array_list, axis=0)

    # Get metadata from one of the input files
    with rio.open(rasters[0]) as src:
        meta = src.meta

    meta.update(dtype=rio.float32)

    with rio.open(output_file, "w", **meta) as dst:
        dst.write(array_out.astype(rio.float32), 1)


def subdaily_to_daily(params, var_type="rootzone"):
    """
    Convert subdaily (3-hourly) soil moisture files to daily averages.
    
    Args:
        params: Configuration parameters object
        var_type: Variable type ('rootzone' or 'surface')
    """
    dir_subdaily = params.dir_intermed / "nsidc" / "subdaily"
    dir_daily = params.dir_intermed / "nsidc" / "daily"

    os.makedirs(dir_daily / var_type, exist_ok=True)

    files = list(Path(dir_subdaily).glob(f"*sm_{var_type}_global.tif"))
    
    if not files:
        params.logger.warning(f"No subdaily files found for {var_type}")
        return

    # Get soil moisture files for a specific date
    df = pd.DataFrame(files, columns=["filepath"])

    # Extract the date from the file name (YYYYDDD format)
    df["timestamp"] = df["filepath"].astype(str).str.extract(r"_(\d{7})T").astype(int)

    # Group by timestamp and get list of files for that timestamp
    df = df.groupby("timestamp")["filepath"].apply(list).reset_index()

    # Loop through each row of the dataframe and average the tif files
    for _, row in tqdm(df.iterrows(), desc=f"subdaily to daily {var_type}", total=len(df)):
        rasters = row["filepath"]

        # Average the rasters and output to disk
        timestamp = str(row["timestamp"])
        year = timestamp[:4]
        doy = timestamp[4:]
        output_file = dir_daily / var_type / f"nasa_usda_soil_moisture_{year}_{doy}_{var_type}_global.tif"

        if os.path.isfile(output_file):
            continue
        else:
            create_daily_sm_file(rasters, output_file)


def process(params):
    """
    Process downloaded H5 files to GeoTIFF format.
    
    Args:
        params: Configuration parameters object
    """
    dir_out = params.dir_download / "nsidc"
    dir_subdaily = params.dir_intermed / "nsidc" / "subdaily"
    os.makedirs(dir_subdaily, exist_ok=True)

    # Convert h5 files to subdaily (3 hourly) GeoTIFFs
    h5_files = list(dir_out.glob("*.h5"))
    params.logger.info(f"Processing {len(h5_files)} H5 files")
    
    for file in tqdm(h5_files, desc="H5 to subdaily TIF"):
        convert(params, file)

    # Convert subdaily data to daily data by averaging
    for var_type in ["rootzone", "surface"]:
        subdaily_to_daily(params, var_type)


def run(params):
    """
    Main entry point for NSIDC SMAP data download and processing.
    
    Args:
        params: Configuration parameters object with:
            - start_year: Start year for data download
            - end_year: End year for data download
            - dir_download: Base download directory
            - dir_intermed: Intermediate data directory
            - logger: Logger instance
    """
    params.logger.info("Starting NSIDC SMAP L4 soil moisture processing")
    
    download(params)
    process(params)
    
    params.logger.info("NSIDC processing complete")


if __name__ == "__main__":
    pass