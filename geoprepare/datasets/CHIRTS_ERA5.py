#!/usr/bin/env python3
"""
Download and reproject CHIRTS-ERA5 daily temperature data.

CHIRTS-ERA5 provides daily maximum (Tmax) and minimum (Tmin) temperature
at 0.05° resolution from CHC/UCSB, using ERA5 reanalysis for temporal
downscaling of the CHIRTSmax/CHIRTSmin monthly climatology.

Format is identical to CHIRPS v3: daily uncompressed GeoTIFFs in year
subdirectories, nodata = -9999.

URLs:
  tmax: https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/tmax/tifs/daily/{year}/
  tmin: https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/tmin/tifs/daily/{year}/

Filename pattern:
  CHIRTS-ERA5.daily_Tmax.{YYYY}.{MM}.{DD}.tif
  CHIRTS-ERA5.daily_Tmin.{YYYY}.{MM}.{DD}.tif

Steps:
  1. Scrape year directory listing and download .tif files.
  2. Scale floating-point °C to integer (×100) and reproject to a global
     3600×7200 grid matching crop masks in a single step.
"""
import logging
import multiprocessing
import os
import time
from calendar import monthrange
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import rasterio
import requests
from bs4 import BeautifulSoup
from osgeo import gdal, osr
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

NODATA_VALUE = -9999
BASE_URL = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5"

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=5, max=80),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch(url, timeout=60):
    """HTTP GET with retry on transient failures."""
    resp = _SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp

# Map config variable name → filename component
_VAR_LABEL = {"tmax": "Tmax", "tmin": "Tmin"}


def get_chirts_url(var, year):
    """Return the directory URL for a given variable and year."""
    return f"{BASE_URL}/{var}/tifs/daily/{year}/"


def get_chirts_filename(var, year, mon, day):
    """Return the remote filename for a given variable and date."""
    label = _VAR_LABEL[var]
    return f"CHIRTS-ERA5.daily_{label}.{year}.{mon:02d}.{day:02d}.tif"


# =========================================================================
# Step 1: Download
# =========================================================================
def download_chirts(var, year, dir_download):
    """Scrape the year directory and download any missing .tif files."""
    out_dir = dir_download / f"chirts_era5/{var}/{year}"
    os.makedirs(out_dir, exist_ok=True)

    # For past years, check if all expected daily files exist locally
    if year < datetime.today().year:
        all_present = True
        for m in range(1, 13):
            for d in range(1, monthrange(year, m)[1] + 1):
                if not (out_dir / get_chirts_filename(var, year, m, d)).exists():
                    all_present = False
                    break
            if not all_present:
                break
        if all_present:
            return

    url = get_chirts_url(var, year)
    logger.info(f"Listing CHIRTS-ERA5 {var} {year} at {url}")

    try:
        resp = _fetch(url, timeout=30)
    except Exception as e:
        logger.error(f"Failed to list {url}: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.select("a[href$='.tif']")

    for link in tqdm(links, desc=f"Download CHIRTS-ERA5 {var} {year}"):
        href = link["href"]
        out_file = out_dir / href.split("/")[-1]
        if not out_file.exists():
            try:
                r2 = _fetch(urljoin(url, href), timeout=120)
                with open(out_file, "wb") as f:
                    f.write(r2.content)
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")
            time.sleep(2)  # throttle between requests


# =========================================================================
# Step 2: Scale + reproject to global grid in one pass
# =========================================================================
def process_to_global(var, year, mon, day, dir_download, dir_intermed, fill_value):
    """Read raw float °C, scale ×100 to int32, place into 3600×7200 global grid."""
    src_path = (
        dir_download / f"chirts_era5/{var}/{year}"
        / get_chirts_filename(var, year, mon, day)
    )
    if not src_path.exists():
        return

    jd = datetime(year, mon, day).timetuple().tm_yday
    global_dir = dir_intermed / f"chirts_era5_{var}" / str(year)
    os.makedirs(global_dir, exist_ok=True)
    out_tif = global_dir / f"chirts_era5_{var}_{year}{jd:03d}_global.tif"

    if out_tif.exists():
        return

    try:
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(float)

        # Mask nodata before scaling to avoid corrupting the sentinel
        nodata_mask = data == NODATA_VALUE
        arr = (data * 100).astype(np.int32)
        arr[nodata_mask] = NODATA_VALUE

        in_height, in_width = arr.shape
        out_arr = np.full((3600, 7200), fill_value, dtype=np.int32)

        if in_height == 2000 and in_width == 7200:
            # 50N to 50S
            out_arr[800:2800, :] = arr
        elif in_height == 2400 and in_width == 7200:
            # 60N to 60S
            out_arr[600:3000, :] = arr
        elif in_height == 3600 and in_width == 7200:
            out_arr = arr
        else:
            lat_offset = (3600 - in_height) // 2
            lon_offset = (7200 - in_width) // 2
            out_arr[
                lat_offset : lat_offset + in_height,
                lon_offset : lon_offset + in_width,
            ] = arr
            logger.warning(f"Non-standard grid size {in_height}x{in_width}")

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(str(out_tif), 7200, 3600, 1, gdal.GDT_Int32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dst.SetProjection(srs.ExportToWkt())
        dst.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
        dst.GetRasterBand(1).SetNoDataValue(fill_value)
        dst.GetRasterBand(1).WriteArray(out_arr)
        dst.FlushCache()

        dst = None
    except Exception as e:
        logger.error(f"Failed to process {src_path}: {e}")


# =========================================================================
# Multiprocessing wrappers
# =========================================================================
def _download_wrapper(args):
    return download_chirts(*args)


def _process_wrapper(args):
    return process_to_global(*args)


# =========================================================================
# Entry point
# =========================================================================
def run(geoprep):
    """Main entry point for CHIRTS-ERA5 processing.

    Args:
        geoprep: GeoDownload object with attributes:
            start_year, end_year, dir_download, dir_intermed,
            redo_last_year, parallel_process, fraction_cpus, fill_value,
            chirts_variables (list of 'tmax'/'tmin').
    """
    start_year = geoprep.start_year
    end_year = geoprep.end_year
    dir_download = Path(geoprep.dir_download)
    dir_intermed = Path(geoprep.dir_intermed)
    parallel_process = geoprep.parallel_process
    fraction_cpus = geoprep.fraction_cpus
    fill_value = geoprep.fill_value

    variables = getattr(geoprep, "chirts_variables", ["tmax", "tmin"])

    num_workers = max(1, int(multiprocessing.cpu_count() * fraction_cpus))

    # Step 1: Download
    download_tasks = [
        (var, yr, dir_download)
        for var in variables
        for yr in range(start_year, end_year + 1)
    ]
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(tqdm(
                p.imap_unordered(_download_wrapper, download_tasks),
                total=len(download_tasks),
                desc="Download CHIRTS-ERA5",
            ))
    else:
        for args in tqdm(download_tasks, desc="Download CHIRTS-ERA5"):
            download_chirts(*args)

    # Step 2: Scale + reproject to global grid
    process_tasks = [
        (var, yr, m, d, dir_download, dir_intermed, fill_value)
        for var in variables
        for yr in range(start_year, end_year + 1)
        for m in range(1, 13)
        for d in range(1, monthrange(yr, m)[1] + 1)
        if (dir_download / f"chirts_era5/{var}/{yr}"
            / get_chirts_filename(var, yr, m, d)).exists()
    ]
    logger.info(f"Processing {len(process_tasks)} CHIRTS-ERA5 files to global grid")
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(tqdm(
                p.imap_unordered(_process_wrapper, process_tasks),
                total=len(process_tasks),
                desc="Process CHIRTS-ERA5",
            ))
    else:
        for args in tqdm(process_tasks, desc="Process CHIRTS-ERA5"):
            process_to_global(*args)
