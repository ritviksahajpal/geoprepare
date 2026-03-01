###############################################################################
# Ritvik Sahajpal
# ritvik@umd.edu
###############################################################################
import os
import glob
import arrow as ar
import numpy as np
from tqdm import tqdm
import requests
from osgeo import osr, gdal
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def delete_existing_files(params, dir_out):
    """

    Args:
        dir_out:

    Returns:

    """
    # Delete current file(s) in CHIRPS-GEFS directory
    filelist = glob.glob(os.path.join(dir_out, "*.tif"))

    for f in tqdm(filelist, desc="Deleting existing CHIRPS-GEFS file(s)"):
        # If today's file is already present, then do not delete
        forecast_regex = "data-mean_" + ar.utcnow().to(
            "America/New_York"
        ).date().strftime("%Y%m%d")
        if forecast_regex in f:
            continue

        # Else delete
        try:
            os.remove(f)
        except:
            params.logger.error("Unable to delete " + f)


def download_CHIRPS_GEFS(params, dir_out):
    base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/"
    MAX_DAYS = 15

    os.makedirs(dir_out, exist_ok=True)
    params.logger.info(f"Downloading to {dir_out}")
    delete_existing_files(params, dir_out)

    for day_offset in range(MAX_DAYS):
        dt = ar.utcnow().to("America/New_York").shift(days=-day_offset)
        download_url = base_url + f"{dt.year}/{dt.month:02d}/{dt.day:02d}"

        response = requests.get(download_url)
        soup = BeautifulSoup(response.text, "html.parser")
        tif_links = soup.select("a[href$='.tif']")

        if tif_links:
            params.logger.info(f"Found {len(tif_links)} files for {dt.format('YYYY-MM-DD')}")
            for link in tqdm(tif_links, desc=f"CHIRPS-GEFS {dt.format('YYYY-M-D')}"):
                filename = os.path.join(dir_out, link["href"].split("/")[-1])
                with open(filename, "wb") as f:
                    f.write(requests.get(urljoin(download_url + "/", link["href"])).content)
            return

        params.logger.info(f"No CHIRPS-GEFS data for {dt.format('YYYY-MM-DD')}, trying previous day")

    params.logger.warning(f"No CHIRPS-GEFS data found in the last {MAX_DAYS} days")


def to_global(params, dir_download):
    """

    Args:
        params ():
        dir_download ():

    Returns:

    """
    # Get path of downloaded CHIRPS-GEFS file
    filelist = glob.glob(os.path.join(dir_download, "*.tif"))
    if len(filelist) != 16:
        params.logger.warning(f"Expected 16 CHIRPS-GEFS files, found {len(filelist)} — skipping")
        return

    current_year = ar.utcnow().to("America/New_York").year
    dir_out = params.dir_intermed / "chirps_gefs" / str(current_year)
    os.makedirs(dir_out, exist_ok=True)
    delete_existing_files(params, dir_out)

    for forecast_file in filelist:
        if not os.path.isfile(dir_out / os.path.basename(forecast_file)):
            fl_out = dir_out / os.path.basename(forecast_file)
            params.logger.info("Creating " + str(fl_out))

            # logger.info(type_product + ' global ' + dir_out / 'chirps_v2.0.' + str(year) + str(jd).zfill(3) + '_global.tif')
            ds = gdal.Open(forecast_file)

            b = ds.GetRasterBand(1)
            bArr = gdal.Band.ReadAsArray(b)
            outArr = np.empty([3600, 7200], dtype=int)
            outArr[0:800, :] = params.fill_value
            outArr[2800:3600, :] = params.fill_value
            outArr[800:2800, :] = bArr * 100  # Multiply by 100 to match CHIRPS format

            otype = gdal.GDT_Int32
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(str(fl_out), 7200, 3600, 1, otype)
            outRasterSRS = osr.SpatialReference()
            outRasterSRS.ImportFromEPSG(4326)
            dst_ds.SetProjection(outRasterSRS.ExportToWkt())
            dst_ds.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
            dst_ds.GetRasterBand(1).WriteArray(outArr)

            ds = None
            dst_ds = None


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    current_year = ar.utcnow().to("America/New_York").year
    dir_download = params.dir_download / "chirps_gefs" / str(current_year)
    os.makedirs(dir_download, exist_ok=True)

    download_CHIRPS_GEFS(params, dir_download)
    to_global(params, dir_download)


if __name__ == "__main__":
    pass
