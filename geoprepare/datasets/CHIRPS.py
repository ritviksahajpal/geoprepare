#!/usr/bin/env python3
"""
Ritvik Sahajpal, Joanne Hall
ritvik@umd.edu

Download, unzip, scale, and reproject CHIRPS data
(preliminary & final) at 0.05° resolution.

Steps:
 1. Download .tif.gz files for each year.
 2. Unzip/rename and convert floating mm to integer (×100)
    with a nodata sentinel.
 3. Reproject to a global 3600×7200 grid matching crop masks.
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


def download_chirps(
    type_data, year, dir_prelim, dir_final, redo_last_year
):
    """
    Download CHIRPS .tif.gz files for a given year and product type.
    """
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    # skip old prelim data if not re-downloading
    if (
        type_data == "prelim"
        and not redo_last_year
        and year < datetime.today().year - 1
    ):
        return

    # set output directory and URL
    if type_data == "prelim":
        out_dir = dir_prelim
        url = f"{base_url}prelim/global_daily/tifs/p05/{year}/"
    else:
        out_dir = dir_final
        url = f"{base_url}global_daily/tifs/p05/{year}/"

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Listing CHIRPS {type_data} {year} at {url}")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to list {url}: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    for link in tqdm(
        soup.select("a[href$='.tif.gz']"),
        desc=f"Download {type_data} {year}",
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
):
    """
    Unzip (if final) or rename (if prelim) and convert to integer-scaled GeoTIFF.
    """
    # prepare paths
    if type_product == "final":
        unzip_dir = dir_interim / "chirps" / type_product / "unzipped"
        os.makedirs(unzip_dir, exist_ok=True)
        zip_path = (
            Path(dir_final)
            / f"chirps-v2.0.{year}.{mon:02d}.{day:02d}.tif.gz"
        )
    else:
        unzip_dir = dir_prelim

    scaled_dir = dir_interim / "chirps" / type_product / "scaled"
    os.makedirs(scaled_dir, exist_ok=True)
    jd = datetime(year, mon, day).timetuple().tm_yday
    out_scaled = (
        scaled_dir / f"chirps_v2_{year}{jd:03d}_scaled.tif"
    )
    if out_scaled.exists():
        return

    # unzip final or use prelim TIFF
    if type_product == "final" and zip_path.exists():
        tif_path = (
            unzip_dir
            / f"chirps-v2.0.{year}.{mon:02d}.{day:02d}.tif"
        )
        logger.info(f"Unzipping {zip_path} to {tif_path}")
        with gzip.open(zip_path, "rb") as zin, open(tif_path, "wb") as zout:
            zout.write(zin.read())
    else:
        tif_path = (
            unzip_dir
            / f"chirps-v2.0.{year}.{mon:02d}.{day:02d}.tif"
        )

    if not tif_path.exists():
        return

    # read and scale
    logger.info(f"Scaling {tif_path} to integer")
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        data = src.read(1).astype(float)

    arr = (data * 100).astype(np.int32)
    arr[data == -999900.0] = NODATA_VALUE
    profile.update(
        dtype=rasterio.int32,
        count=1,
        nodata=NODATA_VALUE,
    )

    with rasterio.open(out_scaled, "w", **profile) as dst:
        dst.write(arr, 1)


def reproject_global(
    type_product,
    year,
    jd,
    dir_interim,
    fill_value,
):
    """
    Reproject scaled TIFF into 3600×7200 global grid.
    """
    scaled_dir = dir_interim / "chirps" / type_product / "scaled"
    global_dir = dir_interim / "chirps" / "global"
    os.makedirs(global_dir, exist_ok=True)

    in_tif = (
        scaled_dir / f"chirps_v2_{year}{jd:03d}_scaled.tif"
    )
    if not in_tif.exists():
        return

    out_tif = (
        global_dir
        / f"chirps_v2.0.{year}{jd:03d}_global.tif"
    )
    if year < datetime.today().year - 1 and out_tif.exists():
        return

    logger.info(f"Reprojecting {in_tif} to {out_tif}")
    ds = gdal.Open(str(in_tif))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    out_arr = np.full((3600, 7200), fill_value, dtype=np.int32)
    out_arr[800:2800, :] = data

    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(out_tif), 7200, 3600, 1, gdal.GDT_Int32
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst.SetProjection(srs.ExportToWkt())
    dst.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
    dst.GetRasterBand(1).WriteArray(out_arr)
    dst.FlushCache()
    ds = None  # close input
    dst = None  # close output


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
    return reproject_global(*args)


def run(
    start_year,
    end_year,
    dir_download,
    dir_interim,
    redo_last_year,
    parallel_process,
    fraction_cpus,
    fill_value,
):
    dir_prelim = dir_download / 'chirps' / 'prelim'
    dir_final = dir_download / 'chirps' / 'final'
    dir_prelim.mkdir(parents=True, exist_ok=True)
    dir_final.mkdir(parents=True, exist_ok=True)

    num_workers = int(multiprocessing.cpu_count() * fraction_cpus)

    # Download
    download_tasks = [
        (prod, yr, dir_prelim, dir_final, redo_last_year)
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
        for args in download_tasks:
            download_chirps(*args)

    # Unzip & Scale
    unzip_tasks = [
        (prod, yr, m, d, dir_prelim, dir_final, dir_interim)
        for prod in LIST_PRODUCTS
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
                    desc="Unzip & Scale",
                )
            )
    else:
        for args in unzip_tasks:
            unzip_and_scale(*args)

    # Reproject
    reproj_tasks = [
        (prod, yr, jd, dir_interim, fill_value)
        for prod in LIST_PRODUCTS
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
        for args in reproj_tasks:
            reproject_global(*args)


if __name__ == "__main__":
    # Example usage; replace arguments as needed
    # run(2020, 2021, Path('/data/download'), Path('/data/interim'), False, False, 0.5, NODATA_VALUE)
    pass