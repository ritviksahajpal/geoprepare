#!/usr/bin/env python3
"""
Download and reproject CHIRPS-GEFS v3 daily precipitation forecasts.

CHIRPS-GEFS v3 provides 16-day global precipitation forecasts at 0.05°.
Each init date has 16 TIF files (today + 15 future days).

URL: https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/daily/global/{year}/{MM}/{DD}/
Filename: c3g_YYYY.MM.DD.tif

Steps:
  1. Scrape latest available init date, download 16 .tif files.
  2. Scale to int (×100) and place into 3600×7200 global grid if needed.
"""
import glob
import logging
import os
from pathlib import Path

import arrow as ar
import numpy as np
import requests
from bs4 import BeautifulSoup
from osgeo import gdal, osr
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/daily/global/"
MAX_LOOKBACK_DAYS = 15

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 geoprepare/CHIRPS-GEFS"
})


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    reraise=True,
)
def _fetch(url, timeout=120):
    """Fetch URL with retry."""
    resp = _SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def download_CHIRPS_GEFS(params, dir_out):
    """Download 16-day CHIRPS-GEFS v3 forecast files."""
    os.makedirs(dir_out, exist_ok=True)
    logger.info(f"Downloading CHIRPS-GEFS v3 to {dir_out}")

    for day_offset in range(MAX_LOOKBACK_DAYS):
        dt = ar.utcnow().to("America/New_York").shift(days=-day_offset)
        url = f"{BASE_URL}{dt.year}/{dt.month:02d}/{dt.day:02d}/"

        try:
            resp = _fetch(url)
        except Exception:
            logger.info(f"No data for {dt.format('YYYY-MM-DD')}, trying previous day")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        tif_links = [a for a in soup.select("a[href$='.tif']")]

        if not tif_links:
            logger.info(f"No .tif files for {dt.format('YYYY-MM-DD')}, trying previous day")
            continue

        logger.info(f"Found {len(tif_links)} files for {dt.format('YYYY-MM-DD')}")
        for link in tqdm(tif_links, desc=f"CHIRPS-GEFS {dt.format('YYYY-MM-DD')}"):
            fname = link["href"].split("/")[-1]
            out_path = Path(dir_out) / fname

            if out_path.exists():
                continue

            try:
                file_resp = _fetch(urljoin(url, link["href"]), timeout=300)
                with open(out_path, "wb") as f:
                    f.write(file_resp.content)
            except Exception as e:
                logger.error(f"Failed to download {fname}: {e}")
                if out_path.exists():
                    out_path.unlink()
        return

    logger.warning(f"No CHIRPS-GEFS data found in the last {MAX_LOOKBACK_DAYS} days")


def to_global(params, dir_download):
    """Scale and reproject downloaded files to standard 3600x7200 global grid."""
    filelist = glob.glob(os.path.join(dir_download, "c3g_*.tif"))
    if not filelist:
        logger.warning("No c3g_*.tif files found — skipping to_global")
        return

    current_year = ar.utcnow().to("America/New_York").year
    dir_out = params.dir_intermed / "chirps_gefs" / str(current_year)
    os.makedirs(dir_out, exist_ok=True)

    for forecast_file in filelist:
        fl_out = dir_out / os.path.basename(forecast_file)
        if fl_out.exists():
            continue

        logger.info(f"Processing {os.path.basename(forecast_file)}")
        try:
            ds = gdal.Open(forecast_file)
            b = ds.GetRasterBand(1)
            bArr = b.ReadAsArray()
            in_height, in_width = bArr.shape

            outArr = np.full([3600, 7200], params.fill_value, dtype=np.int32)

            if in_height == 3600 and in_width == 7200:
                # Already global — just scale
                outArr = (bArr * 100).astype(np.int32)
            elif in_height == 2000 and in_width == 7200:
                # 50N to 50S
                outArr[800:2800, :] = (bArr * 100).astype(np.int32)
            elif in_height == 2400 and in_width == 7200:
                # 60N to 60S
                outArr[600:3000, :] = (bArr * 100).astype(np.int32)
            else:
                # Unknown grid — try to center it
                lat_offset = (3600 - in_height) // 2
                lon_offset = (7200 - in_width) // 2
                outArr[
                    lat_offset:lat_offset + in_height,
                    lon_offset:lon_offset + in_width,
                ] = (bArr * 100).astype(np.int32)
                logger.warning(f"Non-standard grid {in_height}x{in_width}")

            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(str(fl_out), 7200, 3600, 1, gdal.GDT_Int32)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            dst_ds.SetProjection(srs.ExportToWkt())
            dst_ds.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
            dst_ds.GetRasterBand(1).SetNoDataValue(params.fill_value)
            dst_ds.GetRasterBand(1).WriteArray(outArr)
            dst_ds.FlushCache()

            ds = None
            dst_ds = None
        except Exception as e:
            logger.error(f"Failed to process {forecast_file}: {e}")
            if fl_out.exists():
                fl_out.unlink()


def run(params):
    """Main entry point for CHIRPS-GEFS v3 download + processing."""
    current_year = ar.utcnow().to("America/New_York").year
    dir_download = params.dir_download / "chirps_gefs" / str(current_year)
    os.makedirs(dir_download, exist_ok=True)

    download_CHIRPS_GEFS(params, dir_download)
    to_global(params, dir_download)


if __name__ == "__main__":
    pass
