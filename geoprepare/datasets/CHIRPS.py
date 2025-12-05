#!/usr/bin/env python3
"""
Ritvik Sahajpal, Joanne Hall
ritvik@umd.edu

Download, unzip, scale, and reproject CHIRPS data
(preliminary & final) at 0.05° resolution.

Steps:
 1. Download .tif.gz files for each year.
 2. Unzip and convert floating mm to integer (×100) with a nodata sentinel.
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
    Unzip .tif.gz and convert to integer-scaled GeoTIFF.
    Handles both prelim and final data.
    """
    # Set source directory and zip path based on product type
    if type_product == "final":
        src_dir = Path(dir_final)
    else:
        src_dir = Path(dir_prelim)
    
    zip_path = src_dir / f"chirps-v2.0.{year}.{mon:02d}.{day:02d}.tif.gz"
    
    # Set up output directories
    unzip_dir = dir_interim / "chirps" / type_product / "unzipped"
    scaled_dir = dir_interim / "chirps" / type_product / "scaled"
    os.makedirs(unzip_dir, exist_ok=True)
    os.makedirs(scaled_dir, exist_ok=True)
    
    jd = datetime(year, mon, day).timetuple().tm_yday
    out_scaled = scaled_dir / f"chirps_v2_{year}{jd:03d}_scaled.tif"
    
    # Skip if already processed
    if out_scaled.exists():
        return
    
    # Check if source .gz file exists
    if not zip_path.exists():
        return
    
    # Unzip the .tif.gz file
    tif_path = unzip_dir / f"chirps-v2.0.{year}.{mon:02d}.{day:02d}.tif"
    
    if not tif_path.exists():
        logger.info(f"Unzipping {zip_path} to {tif_path}")
        try:
            with gzip.open(zip_path, "rb") as zin, open(tif_path, "wb") as zout:
                zout.write(zin.read())
        except Exception as e:
            logger.error(f"Failed to unzip {zip_path}: {e}")
            return

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
):
    """
    Reproject scaled TIFF into 3600×7200 global grid.
    
    Priority: Final data takes precedence over prelim.
    - If final scaled file exists, use it
    - Otherwise, fall back to prelim
    - If neither exists, skip
    """
    global_dir = dir_interim / "chirps" / "global"
    os.makedirs(global_dir, exist_ok=True)

    out_tif = global_dir / f"chirps_v2.0.{year}{jd:03d}_global.tif"
    # Marker file to track if output was from prelim (so we know to replace with final)
    prelim_marker = global_dir / f".{year}{jd:03d}_from_prelim"
    
    # Check for final and prelim scaled files
    final_scaled = dir_interim / "chirps" / "final" / "scaled" / f"chirps_v2_{year}{jd:03d}_scaled.tif"
    prelim_scaled = dir_interim / "chirps" / "prelim" / "scaled" / f"chirps_v2_{year}{jd:03d}_scaled.tif"
    
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

        out_arr = np.full((3600, 7200), fill_value, dtype=np.int32)
        # CHIRPS native grid is 2000x7200, place it in the correct lat position
        # Native extent: 50N to 50S (rows 800-2800 in 90N to 90S grid)
        out_arr[800:2800, :] = data

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

    dir_prelim = dir_download / 'chirps' / 'prelim'
    dir_final = dir_download / 'chirps' / 'final'
    dir_prelim.mkdir(parents=True, exist_ok=True)
    dir_final.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, int(multiprocessing.cpu_count() * fraction_cpus))

    # =========================================================================
    # Step 1: Download both prelim and final data
    # =========================================================================
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
        for args in tqdm(download_tasks, desc="Download"):
            download_chirps(*args)

    # =========================================================================
    # Step 2: Unzip & Scale - process FINAL first, then PRELIM
    # This ensures final scaled files exist before we do the global reproject
    # =========================================================================
    for prod in ["final", "prelim"]:  # Final first!
        unzip_tasks = [
            (prod, yr, m, d, dir_prelim, dir_final, dir_interim)
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
        (yr, jd, dir_interim, fill_value)
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