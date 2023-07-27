######################################################################################################################
# Ritvik Sahajpal, Jie Zhang
# ritvik@umd.edu
# Data Download
# The python script has been set to only download the grb2 files that do not exist in the folder
# 2 sets of files are being downloaded: as1 (surface) and as2 (subsurface)
# L03 is the folder containing the "real" values and not anomalies
# Units: mm
# Data Preprocessing
# Due to the nature of the grb2 files, a projection and extent first have to be forced onto the newly created tifs
# using gdal_translate. gdal_warp is then used to resample the tif files into 0.05degree global extent tifs
# Ensure the gdal is correctly installed before running these codes.
#######################################################################################################################
import os
import datetime
import glob
import pdb
import subprocess
import multiprocessing
import requests
import re
from bs4 import BeautifulSoup as soup
from tqdm import tqdm
from calendar import monthrange
from pathlib import Path
from osgeo import gdal


def download_soil_moisture(params):
    """

    Args:
        params ():

    Returns:

    """
    dir_download = params.dir_download / "soil_moisture_nasa_usda" / "grib"
    os.makedirs(dir_download, exist_ok=True)

    data_html = soup(requests.get(params.data_dir).text, "lxml")

    # Find all grb2 files on page
    list_links = data_html.findAll(href=re.compile("/*.grb2$"))

    for idx, link in tqdm(
        enumerate(list_links), desc="download soil moisture data", total=len(list_links)
    ):
        # If file has not been downloaded already, then download it
        name_fl = list_links[idx].get("href")

        if not os.path.isfile(dir_download / name_fl):
            download_link = params.data_dir + data_html.findAll(
                href=re.compile("/*.grb2$")
            )[idx].get("href")

            request = requests.head(download_link)

            if request.status_code == 200:
                # params.logger.info('Downloading: ' + name_fl)
                r = requests.get(
                    download_link, os.path.normpath(dir_download / name_fl)
                )
                with open(dir_download / name_fl, "wb") as f:
                    f.write(r.content)


def process_soil_moisture(all_params):
    """

    :param product:
    :param year:
    :param month:
    :param day:
    :return:
    """
    params, product, year, month, day = all_params

    dir_download = params.dir_download / "soil_moisture_nasa_usda" / "grib"
    dir_tif = params.dir_download / "soil_moisture_nasa_usda" / "tif"
    os.makedirs(dir_download, exist_ok=True)
    os.makedirs(dir_tif, exist_ok=True)

    # change to JD
    day_of_year = datetime.date(year, month, day).timetuple().tm_yday

    dir_final = params.dir_interim / f"soil_moisture_{product}"
    os.makedirs(dir_final, exist_ok=True)

    fl_final = f"nasa_usda_soil_moisture_{year}{str(day_of_year).zfill(3)}_{product}_global.tif"

    if not os.path.exists(dir_final / fl_final):
        file_search = glob.glob(
            str(dir_download)
            + os.sep
            + str(year)
            + str(month).zfill(2)
            + str(day).zfill(2)
            + "*"
            + str(product)
            + ".grb2"
        )
        ras_final = os.path.normpath(dir_final / fl_final)

        assert len(file_search) <= 1
        for fl in file_search:
            file = os.path.basename(fl)

            ras_input = os.path.normpath(dir_download / file)

            final_ds = gdal.Warp(
                ras_final,
                ras_input,
                format="GTiff",
                outputType=gdal.GDT_Float32,
                srcNodata=-999.0,
                dstNodata=9999.0,
                srcSRS="EPSG:4326",
                dstSRS="EPSG:4326",
                resampleAlg=gdal.GRA_Bilinear,
                xRes=0.05,
                yRes=0.05,
            )
            final_ds = None


def run(params):
    import itertools

    try:
        download_soil_moisture(params)
    except Exception as e:
        params.logger.error(f"Download of soil moisture data failed {e}")

    all_params = []
    for product in ["as1", "as2"]:
        for year in range(params.start_year, params.end_year + 1):
            for month in range(1, 13):
                for day in range(1, monthrange(year, month)[1] + 1):
                    all_params.extend(
                        list(
                            itertools.product(
                                [params], [product], [year], [month], [day]
                            )
                        )
                    )

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params), desc="process soil moisture data") as pbar:
                for i, _ in tqdm(
                    enumerate(p.imap_unordered(process_soil_moisture, all_params))
                ):
                    pbar.update()
    else:
        pbar = tqdm(
            all_params, total=len(all_params), desc="process soil moisture data"
        )
        for val in pbar:
            pbar.set_description(f"{val[1]} {val[2]} {val[3]} {val[4]}")
            process_soil_moisture(val)
            pbar.update()


if __name__ == "__main__":
    # Surface and Subsurface soil moisture:
    # https://gimms.gsfc.nasa.gov/SMOS/jbolten/FAS/L03/

    # Soil moisture profile:
    # https://gimms.gsfc.nasa.gov/SMOS/jbolten/FAS/L04/

    # Surface and Subsurface soil moisture anomaly:
    # https://gimms.gsfc.nasa.gov/SMOS/jbolten/FAS/L05/
    pass
