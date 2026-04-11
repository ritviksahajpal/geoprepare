#!/usr/bin/env python3
"""
Download, scale, and reproject CHIRTS-ERA5 daily temperature data.

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
  2. Scale floating-point °C to integer (×100) with nodata sentinel.
  3. Reproject to a global 3600×7200 grid matching crop masks.
"""
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

NODATA_VALUE = -9999
BASE_URL = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5"

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
def download_chirts(var, year, dir_download, redo_last_year):
    """Scrape the year directory and download any missing .tif files."""
    if not redo_last_year and year < datetime.today().year - 1:
        return

    out_dir = dir_download / f"chirts_era5/{var}/{year}"
    os.makedirs(out_dir, exist_ok=True)

    url = get_chirts_url(var, year)
    logger.info(f"Listing CHIRTS-ERA5 {var} {year} at {url}")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
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
                r2 = requests.get(urljoin(url, href), timeout=60)
                r2.raise_for_status()
                with open(out_file, "wb") as f:
                    f.write(r2.content)
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")


# =========================================================================
# Step 2: Scale to integer
# =========================================================================
def scale_to_int(var, year, mon, day, dir_download, dir_intermed):
    """Read float °C, multiply by 100, write as int32 GeoTIFF."""
    src_path = (
        dir_download / f"chirts_era5/{var}/{year}"
        / get_chirts_filename(var, year, mon, day)
    )
    if not src_path.exists():
        return

    jd = datetime(year, mon, day).timetuple().tm_yday
    scaled_dir = dir_intermed / f"chirts_era5_{var}" / "scaled" / str(year)
    os.makedirs(scaled_dir, exist_ok=True)
    out_scaled = scaled_dir / f"chirts_era5_{var}_{year}{jd:03d}_scaled.tif"

    if out_scaled.exists():
        return

    try:
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            data = src.read(1).astype(float)

        arr = (data * 100).astype(np.int32)
        arr[data < -9990] = NODATA_VALUE

        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        profile.update(dtype=rasterio.int32, count=1, nodata=NODATA_VALUE)

        with rasterio.open(out_scaled, "w", **profile) as dst:
            dst.write(arr, 1)
    except Exception as e:
        logger.error(f"Failed to scale {src_path}: {e}")


# =========================================================================
# Step 3: Reproject to global grid
# =========================================================================
def reproject_to_global(year, jd, dir_intermed, fill_value, var):
    """Reproject scaled TIFF into 3600×7200 global grid."""
    global_dir = dir_intermed / f"chirts_era5_{var}" / str(year)
    os.makedirs(global_dir, exist_ok=True)

    out_tif = global_dir / f"chirts_era5_{var}_{year}{jd:03d}_global.tif"
    if out_tif.exists():
        return

    scaled_path = (
        dir_intermed / f"chirts_era5_{var}" / "scaled" / str(year)
        / f"chirts_era5_{var}_{year}{jd:03d}_scaled.tif"
    )
    if not scaled_path.exists():
        return

    try:
        ds = gdal.Open(str(scaled_path))
        if ds is None:
            logger.error(f"Failed to open {scaled_path}")
            return

        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        in_height, in_width = data.shape

        out_arr = np.full((3600, 7200), fill_value, dtype=np.int32)

        if in_height == 2000 and in_width == 7200:
            # 50N to 50S
            out_arr[800:2800, :] = data
        elif in_height == 2400 and in_width == 7200:
            # 60N to 60S
            out_arr[600:3000, :] = data
        elif in_height == 3600 and in_width == 7200:
            out_arr = data
        else:
            lat_offset = (3600 - in_height) // 2
            lon_offset = (7200 - in_width) // 2
            out_arr[
                lat_offset : lat_offset + in_height,
                lon_offset : lon_offset + in_width,
            ] = data
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

        ds = None
        dst = None
    except Exception as e:
        logger.error(f"Failed to reproject {scaled_path}: {e}")


# =========================================================================
# Multiprocessing wrappers
# =========================================================================
def _download_wrapper(args):
    return download_chirts(*args)


def _scale_wrapper(args):
    return scale_to_int(*args)


def _reproject_wrapper(args):
    return reproject_to_global(*args)


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
    redo_last_year = geoprep.redo_last_year
    parallel_process = geoprep.parallel_process
    fraction_cpus = geoprep.fraction_cpus
    fill_value = geoprep.fill_value

    variables = getattr(geoprep, "chirts_variables", ["tmax", "tmin"])

    num_workers = max(1, int(multiprocessing.cpu_count() * fraction_cpus))

    # Step 1: Download
    download_tasks = [
        (var, yr, dir_download, redo_last_year)
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

    # Step 2: Scale to integer
    scale_tasks = [
        (var, yr, m, d, dir_download, dir_intermed)
        for var in variables
        for yr in range(start_year, end_year + 1)
        for m in range(1, 13)
        for d in range(1, monthrange(yr, m)[1] + 1)
    ]
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(tqdm(
                p.imap_unordered(_scale_wrapper, scale_tasks),
                total=len(scale_tasks),
                desc="Scale CHIRTS-ERA5",
            ))
    else:
        for args in tqdm(scale_tasks, desc="Scale CHIRTS-ERA5"):
            scale_to_int(*args)

    # Step 3: Reproject to global grid
    reproj_tasks = []
    for var in variables:
        for yr in range(start_year, end_year + 1):
            global_dir = dir_intermed / f"chirts_era5_{var}" / str(yr)
            scaled_dir = dir_intermed / f"chirts_era5_{var}" / "scaled" / str(yr)

            global_files = (
                set(os.listdir(global_dir)) if global_dir.exists() else set()
            )
            scaled_files = (
                set(os.listdir(scaled_dir)) if scaled_dir.exists() else set()
            )

            max_jd = 366 if isleap(yr) else 365
            for jd in range(1, max_jd + 1):
                out_name = f"chirts_era5_{var}_{yr}{jd:03d}_global.tif"
                scaled_name = f"chirts_era5_{var}_{yr}{jd:03d}_scaled.tif"

                if out_name not in global_files and scaled_name in scaled_files:
                    reproj_tasks.append(
                        (yr, jd, dir_intermed, fill_value, var)
                    )

    logger.info(f"Reproject: {len(reproj_tasks)} CHIRTS-ERA5 files need processing")
    if parallel_process:
        with multiprocessing.Pool(num_workers) as p:
            list(tqdm(
                p.imap_unordered(_reproject_wrapper, reproj_tasks),
                total=len(reproj_tasks),
                desc="Reproject CHIRTS-ERA5",
            ))
    else:
        for args in tqdm(reproj_tasks, desc="Reproject CHIRTS-ERA5"):
            reproject_to_global(*args)
