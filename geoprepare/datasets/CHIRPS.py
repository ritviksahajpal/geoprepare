#!/usr/bin/env python3
"""
Ritvik Sahajpal, Joanne Hall
ritvik@umd.edu

Download and reproject CHIRPS data (preliminary & final) at 0.05° resolution.

Supports both CHIRPS v2.0 and v3.0 based on configuration.

CHIRPS v2.0:
 - Files are .tif.gz (gzipped)
 - URL: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{year}/
 - Filename: chirps-v2.0.{year}.{month}.{day}.tif.gz

CHIRPS v3.0:
 - Files are .tif (not gzipped)
 - URL: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/{status}/{disagg}/{year}/
 - Filename: chirps-v3.0.{disagg}.{year}.{month}.{day}.tif
 - Disaggregation options:
   - 'sat': Uses NASA IMERG Late V07 for daily downscaling (available from 1998)
   - 'rnl': Uses ECMWF ERA5 for daily downscaling (full time coverage)
 - Note: Prelim data is only available with 'sat' due to ERA5 latency (5-6 days)

Steps:
 1. Download .tif.gz (v2) or .tif (v3) files for each year.
 2. Scale floating mm to integer (×100), unzip if v2, and reproject to a
    global 3600×7200 grid matching crop masks — all in a single step.

Priority: Final data always takes precedence over prelim data.
When final becomes available, it replaces any existing prelim-based files.
"""
import gzip
import io
import logging
import multiprocessing
import os
import time
from calendar import isleap
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

# Module-level logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

LIST_PRODUCTS = ["prelim", "final"]
NODATA_VALUE = -9999

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


def get_chirps_url(version, type_data, year, disagg=None):
    """
    Get the appropriate URL for CHIRPS data based on version and type.

    Args:
        version: CHIRPS version ('v2' or 'v3')
        type_data: 'prelim' or 'final'
        year: Year to download
        disagg: Disaggregation method for v3 ('sat' or 'rnl').
                Note: prelim only supports 'sat'

    Returns:
        URL string for the data directory
    """
    base_url = "https://data.chc.ucsb.edu/products/"

    if version == "v2":
        if type_data == "prelim":
            return f"{base_url}CHIRPS-2.0/prelim/global_daily/tifs/p05/{year}/"
        else:
            return f"{base_url}CHIRPS-2.0/global_daily/tifs/p05/{year}/"
    else:  # v3
        # Prelim v3 only available with 'sat' disaggregation
        if type_data == "prelim":
            return f"{base_url}CHIRPS/v3.0/daily/prelim/sat/{year}/"
        else:
            return f"{base_url}CHIRPS/v3.0/daily/final/{disagg}/{year}/"


def get_chirps_filename(version, year, mon, day, disagg=None, compressed=True):
    """
    Get the filename for a CHIRPS file based on version.

    Args:
        version: CHIRPS version ('v2' or 'v3')
        year: Year
        mon: Month
        day: Day
        disagg: Disaggregation method for v3 ('sat' or 'rnl')
        compressed: Whether to return .tif.gz (True) or .tif (False)

    Returns:
        Filename string
    """
    if version == "v2":
        base = f"chirps-v2.0.{year}.{mon:02d}.{day:02d}.tif"
        return f"{base}.gz" if compressed else base
    else:  # v3
        # v3 files are never gzipped
        return f"chirps-v3.0.{disagg}.{year}.{mon:02d}.{day:02d}.tif"


def download_chirps(
    type_data, year, dir_prelim, dir_final, redo_last_year, version, disagg
):
    """
    Download CHIRPS .tif.gz (v2) or .tif (v3) files for a given year and product type.

    Args:
        type_data: 'prelim' or 'final'
        year: Year to download
        dir_prelim: Directory for prelim data
        dir_final: Directory for final data
        redo_last_year: Whether to re-download last year's data
        version: CHIRPS version ('v2' or 'v3')
        disagg: Disaggregation method for v3 ('sat' or 'rnl')
    """
    # Skip old prelim data if not re-downloading
    if (
        type_data == "prelim"
        and not redo_last_year
        and year < datetime.today().year - 1
    ):
        return

    # For v3 prelim, only 'sat' is available due to ERA5 latency
    if version == "v3" and type_data == "prelim" and disagg == "rnl":
        logger.info(f"Skipping v3 prelim with rnl - only sat available for prelim")
        return

    # For v3, prelim data only exists for recent years (2025+)
    # Skip prelim for historical years where only final data exists
    if version == "v3" and type_data == "prelim" and year < 2025:
        return

    # Set output directory with year subfolder
    if type_data == "prelim":
        out_dir = dir_prelim / str(year)
    else:
        out_dir = dir_final / str(year)

    os.makedirs(out_dir, exist_ok=True)

    # Get URL for this version/type/year
    url = get_chirps_url(version, type_data, year, disagg)

    logger.info(f"Listing CHIRPS {version} {type_data} {year} at {url}")

    try:
        resp = _fetch(url, timeout=30)
    except Exception as e:
        logger.error(f"Failed to list {url}: {e}")
        return

    # Determine file extension to look for
    file_ext = ".tif.gz" if version == "v2" else ".tif"

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.select(f"a[href$='{file_ext}']")

    for link in tqdm(
        links,
        desc=f"Download {version} {type_data} {year}",
    ):
        href = link["href"]
        out_file = Path(out_dir) / href.split("/")[-1]
        if not out_file.exists():
            try:
                r2 = _fetch(urljoin(url, href), timeout=120)
                with open(out_file, "wb") as f:
                    f.write(r2.content)
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")
            time.sleep(2)  # throttle between requests


def _read_chirps_tif(src_path, version):
    """Read a CHIRPS file (handles v2 gzip decompression) and return float data.

    Args:
        src_path: Path to the .tif (v3) or .tif.gz (v2) file
        version: 'v2' or 'v3'

    Returns:
        numpy float array, or None on failure
    """
    try:
        if version == "v2":
            raw = gzip.decompress(src_path.read_bytes())
            with rasterio.open(io.BytesIO(raw)) as src:
                return src.read(1).astype(float)
        else:
            with rasterio.open(src_path) as src:
                return src.read(1).astype(float)
    except Exception as e:
        logger.error(f"Failed to read {src_path}: {e}")
        return None


def process_to_global(
    year,
    mon,
    day,
    dir_prelim,
    dir_final,
    dir_intermed,
    fill_value,
    version,
    disagg,
):
    """
    Read raw CHIRPS file, scale ×100 to int32, and write to 3600×7200 global grid.

    Final data takes precedence over prelim. A '.from_prelim' marker file tracks
    whether the global output came from prelim, so it can be replaced when final
    data becomes available.

    Args:
        year, mon, day: Date
        dir_prelim: Directory for prelim downloads
        dir_final: Directory for final downloads
        dir_intermed: Directory for output global files
        fill_value: NoData fill value for global grid
        version: 'v2' or 'v3'
        disagg: Disaggregation method for v3
    """
    jd = datetime(year, mon, day).timetuple().tm_yday
    version_str = "v2.0" if version == "v2" else "v3.0"

    global_dir = dir_intermed / "chirps" / version / "global" / str(year)
    os.makedirs(global_dir, exist_ok=True)

    out_tif = global_dir / f"chirps_{version_str}_{year}{jd:03d}_global.tif"
    prelim_marker = global_dir / f".{year}{jd:03d}_from_prelim"

    # Locate source files — final preferred over prelim
    if version == "v2":
        final_path = dir_final / str(year) / get_chirps_filename(version, year, mon, day, disagg, compressed=True)
        prelim_path = dir_prelim / str(year) / get_chirps_filename(version, year, mon, day, disagg, compressed=True)
    else:
        actual_disagg_final = disagg
        actual_disagg_prelim = "prelim"
        final_path = dir_final / str(year) / get_chirps_filename(version, year, mon, day, actual_disagg_final, compressed=False)
        prelim_path = dir_prelim / str(year) / get_chirps_filename(version, year, mon, day, actual_disagg_prelim, compressed=False)

    # Determine which source to use
    if final_path.exists():
        src_path = final_path
        use_prelim = False
        if prelim_marker.exists():
            # Final now available — replace prelim output
            logger.info(f"Replacing prelim with final for {year} DOY {jd}")
            prelim_marker.unlink()
        elif out_tif.exists():
            # Already processed from final, skip
            return
    elif prelim_path.exists():
        src_path = prelim_path
        use_prelim = True
        if out_tif.exists():
            return
    else:
        return

    data = _read_chirps_tif(src_path, version)
    if data is None:
        return

    # Scale mm to integer (×100) and set nodata
    arr = (data * 100).astype(np.int32)
    arr[data < -9990] = NODATA_VALUE

    in_height, in_width = arr.shape
    out_arr = np.full((3600, 7200), fill_value, dtype=np.int32)

    if in_height == 2000 and in_width == 7200:
        # CHIRPS v2: 50N to 50S → rows 800-2800
        out_arr[800:2800, :] = arr
    elif in_height == 2400 and in_width == 7200:
        # CHIRPS v3: 60N to 60S → rows 600-3000
        out_arr[600:3000, :] = arr
    elif in_height == 3600 and in_width == 7200:
        out_arr = arr
    else:
        lat_offset = (3600 - in_height) // 2
        lon_offset = (7200 - in_width) // 2
        out_arr[lat_offset:lat_offset + in_height, lon_offset:lon_offset + in_width] = arr
        logger.warning(f"Non-standard grid size {in_height}x{in_width}")

    try:
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

        if use_prelim:
            prelim_marker.touch()
    except Exception as e:
        logger.error(f"Failed to write {out_tif}: {e}")


# ============================================================================
# Wrapper functions for multiprocessing
# ============================================================================
def _download_chirps_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return download_chirps(*args)


def _process_to_global_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return process_to_global(*args)


def run(geoprep):
    """
    Main entry point for CHIRPS processing.

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
            - fill_value: NoData fill value
            - version: CHIRPS version ('v2' or 'v3')
            - disagg: Disaggregation method for v3 ('sat' or 'rnl')
    """
    # Extract parameters from geoprep object
    start_year = geoprep.start_year
    end_year = geoprep.end_year
    dir_download = Path(geoprep.dir_download)
    dir_intermed = Path(geoprep.dir_intermed)
    redo_last_year = geoprep.redo_last_year
    parallel_process = geoprep.parallel_process
    fraction_cpus = geoprep.fraction_cpus
    fill_value = geoprep.fill_value

    # Get version and disaggregation method (with defaults for backward compatibility)
    version = getattr(geoprep, 'version', 'v2')
    disagg = getattr(geoprep, 'disagg', 'sat')

    # Validate version
    if version not in ['v2', 'v3']:
        raise ValueError(f"Invalid CHIRPS version '{version}'. Must be 'v2' or 'v3'.")

    # Validate disaggregation method for v3
    if version == 'v3' and disagg not in ['sat', 'rnl']:
        raise ValueError(f"Invalid disaggregation method '{disagg}'. Must be 'sat' or 'rnl'.")

    logger.info(f"Processing CHIRPS {version}" + (f" with {disagg} disaggregation" if version == 'v3' else ""))

    # Set up directories - include version in path
    dir_prelim = dir_download / 'chirps' / version / 'prelim'
    dir_final = dir_download / 'chirps' / version / 'final'
    dir_prelim.mkdir(parents=True, exist_ok=True)
    dir_final.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, int(multiprocessing.cpu_count() * fraction_cpus))

    # =========================================================================
    # Step 1: Download both prelim and final data
    # =========================================================================
    download_tasks = [
        (prod, yr, dir_prelim, dir_final, redo_last_year, version, disagg)
        for prod in LIST_PRODUCTS
        for yr in range(start_year, end_year + 1)
    ]
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(
                tqdm(
                    p.imap_unordered(_download_chirps_wrapper, download_tasks),
                    total=len(download_tasks),
                    desc="Download",
                )
            )
    else:
        for args in tqdm(download_tasks, desc="Download"):
            download_chirps(*args)

    # =========================================================================
    # Step 2: Scale + reproject to global grid (final priority over prelim)
    # =========================================================================
    version_str = "v2.0" if version == "v2" else "v3.0"
    process_tasks = []
    for yr in range(start_year, end_year + 1):
        global_dir = dir_intermed / "chirps" / version / "global" / str(yr)
        global_files = set(os.listdir(global_dir)) if global_dir.exists() else set()

        # Collect downloaded files per product type for fast lookup
        final_dir = dir_final / str(yr)
        prelim_dir = dir_prelim / str(yr)
        final_files = set(os.listdir(final_dir)) if final_dir.exists() else set()
        prelim_files = set(os.listdir(prelim_dir)) if prelim_dir.exists() else set()

        max_jd = 366 if isleap(yr) else 365
        for jd in range(1, max_jd + 1):
            out_name = f"chirps_{version_str}_{yr}{jd:03d}_global.tif"
            marker_name = f".{yr}{jd:03d}_from_prelim"
            out_exists = out_name in global_files

            # Determine date from jd
            dt = datetime.strptime(f"{yr}{jd:03d}", "%Y%j")
            mon, day = dt.month, dt.day

            # Check which source files exist
            if version == "v2":
                final_fname = get_chirps_filename(version, yr, mon, day, disagg, compressed=True)
                prelim_fname = final_fname  # same filename pattern, different directory
            else:
                final_fname = get_chirps_filename(version, yr, mon, day, disagg, compressed=False)
                prelim_fname = get_chirps_filename(version, yr, mon, day, "prelim", compressed=False)

            has_final = final_fname in final_files
            has_prelim = prelim_fname in prelim_files

            needs_processing = False
            if not out_exists and (has_final or has_prelim):
                needs_processing = True
            elif out_exists and marker_name in global_files and has_final:
                # Prelim output exists but final is now available — replace
                needs_processing = True

            if needs_processing:
                process_tasks.append(
                    (yr, mon, day, dir_prelim, dir_final, dir_intermed,
                     fill_value, version, disagg)
                )

    logger.info(f"Processing {len(process_tasks)} CHIRPS files to global grid")
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(
                tqdm(
                    p.imap_unordered(_process_to_global_wrapper, process_tasks),
                    total=len(process_tasks),
                    desc="Process CHIRPS",
                )
            )
    else:
        for args in tqdm(process_tasks, desc="Process CHIRPS"):
            process_to_global(*args)


if __name__ == "__main__":
    pass
