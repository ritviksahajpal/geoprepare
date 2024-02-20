###############################################################################
# Ritvik Sahajpal
# ritvik@umd.edu
###############################################################################
import os
import pdb
import glob
import arrow as ar
import numpy as np
from tqdm import tqdm
import requests
from osgeo import osr, gdal
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_forecast_file_name(params, list_files):
    """

    Args:
        list_files:

    Returns:

    """
    # Number of days to look back at if forecast file does not exist for current day
    MAX_DAYS = 15

    forecast_regex = "data-mean_" + ar.utcnow().to("America/New_York").date().strftime(
        "%Y%m%d"
    )
    forecast_file = [s for s in list_files if forecast_regex in s]

    # If forecast file does not exist for current day then look at previous day (upto 14 days ago)
    if not len(forecast_file):
        params.logger.info(
            "CHIRPS-GEFS data does not exist for "
            + ar.utcnow().to("America/New_York").date().strftime("%Y-%m-%d")
        )
        for day in range(1, MAX_DAYS):
            forecast_regex = "data-mean_" + ar.utcnow().to("America/New_York").shift(
                days=-day
            ).date().strftime("%Y%m%d")
            forecast_file = [s for s in list_files if forecast_regex in s]

            # If data exists then break
            if len(forecast_file):
                params.logger.info(
                    "Getting CHIRPS-GEFS data from "
                    + ar.utcnow()
                    .to("America/New_York")
                    .shift(days=-day)
                    .date()
                    .strftime("%Y-%m-%d")
                )
                break

    return forecast_file


def get_forecast_file_name(params, list_files):
    """

    Args:
        list_files:

    Returns:

    """
    # Number of days to look back at if forecast file does not exist for current day
    MAX_DAYS = 15
    current_day = ar.utcnow().date().strftime("%Y%m%d")

    forecast_regex = f"data-mean_{current_day}"
    forecast_file = [s for s in list_files if forecast_regex in s]

    # If forecast file does not exist for current day then look at previous day (upto 14 days ago)
    if not len(forecast_file):
        params.logger.info(f"CHIRPS-GEFS data does not exist for date: {current_day}")

        for day in range(1, MAX_DAYS):
            previous_day = ar.utcnow().shift(days=-day).date().strftime("%Y%m%d")

            forecast_regex = f"data-mean_{previous_day}"
            forecast_file = [s for s in list_files if forecast_regex in s]

            # If data exists then break
            if len(forecast_file):
                params.logger.info(f"Getting CHIRPS-GEFS data from {previous_day}")
                break

    return forecast_file


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

    current_year = ar.utcnow().to("America/New_York").year
    current_month = ar.utcnow().to("America/New_York").month
    current_day = ar.utcnow().to("America/New_York").day
    download_url = base_url + f"{current_year}/{current_month:02d}/{current_day:02d}"

    os.makedirs(dir_out, exist_ok=True)
    params.logger.info(f"Downloading to {dir_out}")
    delete_existing_files(params, dir_out)

    response = requests.get(download_url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in tqdm(
        soup.select("a[href$='.tif']"),
        desc=f"CHIRPS-GEFS {current_year}-{current_month}-{current_day}",
    ):
        # Name the pdf files using the last portion of each link which are unique in this case
        filename = os.path.join(dir_out, link["href"].split("/")[-1])

        with open(filename, "wb") as f:
            f.write(requests.get(urljoin(download_url + "/", link["href"])).content)


def to_global(params, dir_download):
    """

    Args:
        params ():
        dir_download ():

    Returns:

    """
    # Get path of downloaded CHIRPS-GEFS file
    filelist = glob.glob(os.path.join(dir_download, "*.tif"))
    assert len(filelist) == 16

    dir_out = params.dir_interim / "chirps_gefs"
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
    dir_download = params.dir_download / "chirps_gefs"
    os.makedirs(dir_download, exist_ok=True)

    download_CHIRPS_GEFS(params, dir_download)
    to_global(params, dir_download)


if __name__ == "__main__":
    pass
