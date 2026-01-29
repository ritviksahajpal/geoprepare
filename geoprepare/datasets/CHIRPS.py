#!/usr/bin/env python3
"""
Ritvik Sahajpal, Joanne Hall
ritvik@umd.edu

Download, unzip (if needed), scale, and reproject CHIRPS data
(preliminary & final) at 0.05° resolution.

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
 2. Unzip (v2 only) and convert floating mm to integer (×100) with a nodata sentinel.
 3. Reproject to a global 3600×7200 grid matching crop masks.
 
Priority: Final data always takes precedence over prelim data.
When final becomes available, it replaces any existing prelim-based files.
"""
import gzip
import logging
import multiprocessing
import os
from calendar import isleap, monthrange
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import rasterio
import requests
from bs4 import BeautifulSoup
from osgeo import gdal, osr
from tqdm import tqdm

# Module-level logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

LIST_PRODUCTS = ["prelim", "final"]
NODATA_VALUE = -9999


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

    # Set output directory
    if type_data == "prelim":
        out_dir = dir_prelim
    else:
        out_dir = dir_final

    os.makedirs(out_dir, exist_ok=True)
    
    # Get URL for this version/type/year
    url = get_chirps_url(version, type_data, year, disagg)
    
    logger.info(f"Listing CHIRPS {version} {type_data} {year} at {url}")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to list {url}: {e}")
        return

    # Determine file extension to look for
    if version == "v2":
        file_ext = ".tif.gz"
    else:
        file_ext = ".tif"

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
                r2 = requests.get(urljoin(url, href), timeout=60)
                r2.raise_for_status()
                with open(out_file, "wb") as f:
                    f.write(r2.content)
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")


def unzip_and_scale(
    type_product,
    year,
    mon,
    day,
    dir_prelim,
    dir_final,
    dir_interim,
    version,
    disagg,
):
    """
    Unzip .tif.gz (v2 only) and convert to integer-scaled GeoTIFF.
    Handles both prelim and final data for v2 and v3.
    
    Args:
        type_product: 'prelim' or 'final'
        year: Year
        mon: Month
        day: Day
        dir_prelim: Directory for prelim data
        dir_final: Directory for final data
        dir_interim: Directory for interim processed data
        version: CHIRPS version ('v2' or 'v3')
        disagg: Disaggregation method for v3 ('sat' or 'rnl')
    """
    # For v3 prelim, only 'sat' is available
    if version == "v3" and type_product == "prelim" and disagg == "rnl":
        return
    
    # Set source directory based on product type
    if type_product == "final":
        src_dir = Path(dir_final)
    else:
        src_dir = Path(dir_prelim)
    
    # Get source filename based on version
    if version == "v2":
        src_filename = get_chirps_filename(version, year, mon, day, disagg, compressed=True)
        src_path = src_dir / src_filename
        is_compressed = True
    else:  # v3
        # For v3 prelim, always use 'sat' regardless of disagg setting
        actual_disagg = "sat" if type_product == "prelim" else disagg
        src_filename = get_chirps_filename(version, year, mon, day, actual_disagg, compressed=False)
        src_path = src_dir / src_filename
        is_compressed = False
    
    # Set up output directories - include version in path
    unzip_dir = dir_interim / "chirps" / version / type_product / "unzipped"
    scaled_dir = dir_interim / "chirps" / version / type_product / "scaled"
    os.makedirs(unzip_dir, exist_ok=True)
    os.makedirs(scaled_dir, exist_ok=True)
    
    jd = datetime(year, mon, day).timetuple().tm_yday
    version_str = "v2.0" if version == "v2" else "v3.0"
    out_scaled = scaled_dir / f"chirps_{version_str}_{year}{jd:03d}_scaled.tif"
    
    # Skip if already processed
    if out_scaled.exists():
        return
    
    # Check if source file exists
    if not src_path.exists():
        return
    
    # Handle unzipping for v2 (compressed) files
    if is_compressed:
        tif_filename = get_chirps_filename(version, year, mon, day, disagg, compressed=False)
        tif_path = unzip_dir / tif_filename
        
        if not tif_path.exists():
            logger.info(f"Unzipping {src_path} to {tif_path}")
            try:
                with gzip.open(src_path, "rb") as zin, open(tif_path, "wb") as zout:
                    zout.write(zin.read())
            except Exception as e:
                logger.error(f"Failed to unzip {src_path}: {e}")
                return
    else:
        # v3 files are not compressed, use directly
        tif_path = src_path

    # Read and scale
    logger.info(f"Scaling {tif_path} to integer")
    try:
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            data = src.read(1).astype(float)

        arr = (data * 100).astype(np.int32)
        # Handle CHIRPS nodata value (-9999 in raw, becomes -999900 after *100)
        arr[data < -9990] = NODATA_VALUE
        profile.update(
            dtype=rasterio.int32,
            count=1,
            nodata=NODATA_VALUE,
        )

        with rasterio.open(out_scaled, "w", **profile) as dst:
            dst.write(arr, 1)
    except Exception as e:
        logger.error(f"Failed to scale {tif_path}: {e}")


def reproject_to_global(
    year,
    jd,
    dir_interim,
    fill_value,
    version,
):
    """
    Reproject scaled TIFF into 3600×7200 global grid.
    
    Priority: Final data takes precedence over prelim.
    - If final scaled file exists, use it
    - Otherwise, fall back to prelim
    - If neither exists, skip
    
    Args:
        year: Year
        jd: Julian day (day of year)
        dir_interim: Directory for interim processed data
        fill_value: Fill value for nodata
        version: CHIRPS version ('v2' or 'v3')
    """
    global_dir = dir_interim / "chirps" / version / "global"
    os.makedirs(global_dir, exist_ok=True)

    version_str = "v2.0" if version == "v2" else "v3.0"
    out_tif = global_dir / f"chirps_{version_str}_{year}{jd:03d}_global.tif"
    # Marker file to track if output was from prelim (so we know to replace with final)
    prelim_marker = global_dir / f".{year}{jd:03d}_from_prelim"
    
    # Check for final and prelim scaled files
    final_scaled = dir_interim / "chirps" / version / "final" / "scaled" / f"chirps_{version_str}_{year}{jd:03d}_scaled.tif"
    prelim_scaled = dir_interim / "chirps" / version / "prelim" / "scaled" / f"chirps_{version_str}_{year}{jd:03d}_scaled.tif"
    
    # Determine which source to use
    if final_scaled.exists():
        in_tif = final_scaled
        use_prelim = False
        # If we have final data and prelim marker exists, replace prelim with final
        if prelim_marker.exists():
            logger.info(f"Replacing prelim with final for {year} DOY {jd}")
            prelim_marker.unlink()
        elif out_tif.exists():
            # Already processed from final, skip
            return
    elif prelim_scaled.exists():
        in_tif = prelim_scaled
        use_prelim = True
        # If prelim output already exists, skip
        if out_tif.exists():
            return
    else:
        # Neither exists
        return

    logger.info(f"Reprojecting {in_tif} to {out_tif}")
    
    try:
        ds = gdal.Open(str(in_tif))
        if ds is None:
            logger.error(f"Failed to open {in_tif}")
            return
            
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # Get input dimensions to handle different grid sizes
        in_height, in_width = data.shape

        out_arr = np.full((3600, 7200), fill_value, dtype=np.int32)
        
        # CHIRPS v2 native grid is 2000x7200 (50N to 50S at 0.05°)
        # CHIRPS v3 may have different dimensions - handle accordingly
        if in_height == 2000 and in_width == 7200:
            # Standard CHIRPS grid: place in rows 800-2800 (50N to 50S in 90N to 90S grid)
            out_arr[800:2800, :] = data
        elif in_height == 3600 and in_width == 7200:
            # Already global grid
            out_arr = data
        else:
            # Handle other grid sizes by computing offset
            # Assuming 0.05° resolution centered on equator
            lat_offset = (3600 - in_height) // 2
            lon_offset = (7200 - in_width) // 2
            out_arr[lat_offset:lat_offset + in_height, lon_offset:lon_offset + in_width] = data
            logger.warning(f"Non-standard grid size {in_height}x{in_width}, placed with offset")

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(
            str(out_tif), 7200, 3600, 1, gdal.GDT_Int32
        )
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dst.SetProjection(srs.ExportToWkt())
        dst.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
        dst.GetRasterBand(1).SetNoDataValue(fill_value)
        dst.GetRasterBand(1).WriteArray(out_arr)
        dst.FlushCache()
        
        # Clean up
        ds = None
        dst = None
        
        # Create prelim marker if using prelim data
        if use_prelim:
            prelim_marker.touch()
            
    except Exception as e:
        logger.error(f"Failed to reproject {in_tif}: {e}")


# ============================================================================
# Wrapper functions for multiprocessing
# ============================================================================
def _download_chirps_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return download_chirps(*args)


def _unzip_and_scale_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return unzip_and_scale(*args)


def _reproject_global_wrapper(args):
    """Wrapper to unpack tuple args for multiprocessing."""
    return reproject_to_global(*args)


def run(geoprep):
    """
    Main entry point for CHIRPS processing.
    
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
            - fill_value: NoData fill value
            - version: CHIRPS version ('v2' or 'v3')
            - disagg: Disaggregation method for v3 ('sat' or 'rnl')
    """
    # Extract parameters from geoprep object
    start_year = geoprep.start_year
    end_year = geoprep.end_year
    dir_download = Path(geoprep.dir_download)
    dir_interim = Path(geoprep.dir_interim)
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
    # Step 2: Unzip & Scale - process FINAL first, then PRELIM
    # This ensures final scaled files exist before we do the global reproject
    # =========================================================================
    for prod in ["final", "prelim"]:  # Final first!
        unzip_tasks = [
            (prod, yr, m, d, dir_prelim, dir_final, dir_interim, version, disagg)
            for yr in range(start_year, end_year + 1)
            for m in range(1, 13)
            for d in range(1, monthrange(yr, m)[1] + 1)
        ]
        if parallel_process:
            with multiprocessing.Pool(num_workers) as p:
                list(
                    tqdm(
                        p.imap_unordered(_unzip_and_scale_wrapper, unzip_tasks),
                        total=len(unzip_tasks),
                        desc=f"Unzip & Scale ({prod})",
                    )
                )
        else:
            for args in tqdm(unzip_tasks, desc=f"Unzip & Scale ({prod})"):
                unzip_and_scale(*args)

    # =========================================================================
    # Step 3: Reproject to global grid
    # The reproject function handles final vs prelim priority internally
    # =========================================================================
    reproj_tasks = [
        (yr, jd, dir_interim, fill_value, version)
        for yr in range(start_year, end_year + 1)
        for jd in range(1, (366 if isleap(yr) else 365) + 1)
    ]
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(
                tqdm(
                    p.imap_unordered(_reproject_global_wrapper, reproj_tasks),
                    total=len(reproj_tasks),
                    desc="Reproject",
                )
            )
    else:
        for args in tqdm(reproj_tasks, desc="Reproject"):
            reproject_to_global(*args)


if __name__ == "__main__":
    pass