########################################################################################################################
# Ritvik Sahajpal, Joanne Hall
# ritvik@umd.edu
#
# The original data is in zipped/unzipped tiff format at 0.05 degree resolution (zipped for final products and unzipped
# for preliminary product in recent years). The naming convention is in year, month, day and to match the other datasets
# the data has to be renamed into year, julian day.
#
# Final Tiffs
# Step 1: Unzip, rename the original tiff files, and convert the floating point (unit: mm) data into integer by scaling
# by 100. This speeds up step 2 and the Weighted Average Extraction code.
# Step 2: Convert the data into a global extent to match the crop masks.
#
# Preliminary Tiffs
# Step 1: Rename the original tiff files. Convert the floating point (unit: mm) data into integer by scaling by 100.
# This speeds up step 2 and the Weighted Average Extraction code.
# Step 2: Convert the data into a global extent to match the crop masks.
########################################################################################################################
import os
import pdb
import requests
import rasterio
import itertools
import multiprocessing
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from pathlib import Path

list_products = ["prelim", "final"]


def download_CHIRPS(all_params):
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    type_data, year, params, dir_prelim, dir_final = all_params

    # prelim data should only be downloaded for current and previous year at most
    if type_data == 'prelim' and (params.redo_last_year and year < datetime.today().year - 1):
        return

    if type_data == 'prelim':
        dir_out = dir_prelim
        download_url = base_url + "prelim/global_daily/tifs/p05/" + str(year)
    elif type_data == 'final':
        dir_out = dir_final
        download_url = base_url + "global_daily/tifs/p05/" + str(year)

    os.makedirs(dir_out, exist_ok=True)
    print(f'Downloading to {dir_out}')

    response = requests.get(download_url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in tqdm(soup.select("a[href$='.tif.gz']"), desc=f'CHIRPS {type_data} {year}'):
        # Name the pdf files using the last portion of each link which are unique in this case
        filename = os.path.join(dir_out, link['href'].split('/')[-1])

        if not os.path.isfile(filename):
            with open(filename, 'wb') as f:
                f.write(requests.get(urljoin(download_url + "/", link['href'])).content)

def download(all_params):
    """
    Preliminary CHIRPS data: /pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/
    Final CHIRPS data: /pub/org/chc/products/CHIRPS-2.0/global_daily/tifs/p05/
    Download CHIRPS preliminary and final files
    store prelim in: /download/chirps/prelim
    store final in: /download/chirps/final.

    Args:
        all_params ():

    Returns:

    """
    from ftplib import FTP

    type_data, year, params, dir_prelim, dir_final = all_params

    # prelim data should only be downloaded for current and previous year at most
    if type_data == "prelim" and (
        params.redo_last_year and year < datetime.today().year - 1
    ):
        return

    with FTP("ftp.chc.ucsb.edu", "anonymous") as ftp:
        if type_data == "prelim":
            try:
                ftp.cwd(params.prelim + str(year))
            except Exception as e:
                params.logger.error(f"Could not connect to FTP server: {e}")
                return SystemError(f"Encountered error {e}")
            dir_out = dir_prelim

        if type_data == "final":
            ftp.cwd(params.final + str(year))
            dir_out = dir_final

        os.makedirs(dir_out, exist_ok=True)

        list_files = ftp.nlst()

        for fl in tqdm(list_files, desc=f"Downloading {type_data} CHIRPS"):
            if fl.find("chirps") > -1:
                local_filename = os.path.join(dir_out, fl)

                if os.path.isfile(dir_out / fl):
                    pass
                else:
                    with open(local_filename, "wb") as lf:
                        ftp.retrbinary("RETR " + fl, lf.write)


def unzip(all_params):
    """
    Store CHIRPS preliminary unzipped (/interim/chirps/prelim/unzipped)
    Store CHIRPS final scaled (/interim/chirps/final/scaled)
    Store CHIRPS prelim scaled (/interim/chirps/prelim/scaled)
    :param all_params:
    :return:
    """
    import gzip

    type_product, year, mon, day, params, dir_prelim, dir_final = all_params

    if type_product == "final":
        dir_unzipped = params.dir_interim / "chirps" / type_product / "unzipped"
        os.makedirs(dir_unzipped, exist_ok=True)

        # fl_zip refers to the downloaded zipped CHIRPS final file
        fl_zip = dir_final / Path(
            f"chirps-v2.0.{year}.{str(mon).zfill(2)}.{str(day).zfill(2)}.tif.gz"
        )

    dir_integer = params.dir_interim / "chirps" / type_product / "scaled"
    os.makedirs(dir_integer, exist_ok=True)

    # change to JD
    day_of_year = datetime(year, mon, day).timetuple().tm_yday

    # fl_int is the unzipped and scaled file we create
    fl_int = dir_integer / Path(
        f"chirps_v2_{year}{str(day_of_year).zfill(3)}_scaled.tif"
    )

    if os.path.exists(fl_int):
        params.logger.info(f"File exists {fl_int}")
    else:
        if type_product == "final":
            fl_unzip = dir_unzipped / Path(
                f"chirps_v2.0.{year}{str(day_of_year).zfill(3)}_unzip.tif"
            )

            if os.path.isfile(fl_zip):
                params.logger.info(f"Unzipping {fl_zip} to {fl_unzip}")
                with gzip.open(fl_zip, "rb") as hndl_zip:
                    with open(fl_unzip, "wb") as hndl_unzip:
                        hndl_unzip.write(hndl_zip.read())
        else:
            fl_unzip = dir_prelim / Path(
                f"chirps-v2.0.{year}.{str(mon).zfill(2)}.{str(day).zfill(2)}.tif"
            )

        if os.path.isfile(fl_unzip):
            params.logger.info(f"Converting to int {fl_unzip} to {fl_int}")
            with rasterio.open(os.path.normpath(fl_unzip)) as dataset:
                profile = dataset.profile
                profile.update(dtype=rasterio.int32, count=1)
                data = dataset.read()

            arr = data * 100  # multiply by 100 to convert to integer
            arr[arr == -999900.0] = None  # Assigning null value

            with rasterio.open(fl_int, "w", **profile) as dst:
                dst.write(arr.astype(rasterio.int32))


def to_global(all_params):
    """
    Store CHIRPS preliminary global (/interim/chirps/prelim/global)
    Args:
        all_params ():

    Returns:

    """
    from osgeo import osr, gdal

    type_product, year, jd, params = all_params

    dir_integer = params.dir_interim / "chirps" / type_product / "scaled"
    dir_out = params.dir_interim / "chirps" / "global"
    os.makedirs(dir_out, exist_ok=True)

    if os.path.isfile(
        dir_integer / Path(f"chirps_v2_{year}{str(jd).zfill(3)}_scaled.tif")
    ):
        fl_out = dir_out / Path(f"chirps_v2.0.{year}{str(jd).zfill(3)}_global.tif")

        # Redo output file if it already exists, if it is this year or last
        if year < (datetime.today().year - 1) and os.path.isfile(fl_out):
            return

        params.logger.info(
            f"{type_product} global {dir_out} / chirps_v2.0.{year}{str(jd).zfill(3)}_global.tif"
        )
        ds = gdal.Open(
            str(dir_integer) + os.sep + f"chirps_v2_{year}{str(jd).zfill(3)}_scaled.tif"
        )

        b = ds.GetRasterBand(1)
        bArr = gdal.Band.ReadAsArray(b)
        outArr = np.empty([3600, 7200], dtype=int)
        outArr[0:800, :] = params.fill_value
        outArr[2800:3600, :] = params.fill_value
        outArr[800:2800, :] = bArr

        otype = gdal.GDT_Int32
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(str(fl_out), 7200, 3600, 1, otype)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
        dst_ds.GetRasterBand(1).WriteArray(outArr)

        ds = None


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    import calendar
    from calendar import monthrange

    start_year, end_year = params.start_year, params.end_year + 1
    parallel_process = params.parallel_process
    fraction_cpus = params.fraction_cpus
    dir_prelim = (
        params.dir_download / "chirps" / "prelim"
    )  # unzipped tiff of CHIRPS prelim
    dir_final = params.dir_download / "chirps" / "final"  # zipped tiff of CHIRPS final

    os.makedirs(dir_prelim, exist_ok=True)
    os.makedirs(dir_final, exist_ok=True)

    ##########
    # DOWNLOAD
    ##########
    all_params = []
    for type_data in list_products:
        for year in range(start_year, end_year):
            all_params.extend(
                list(
                    itertools.product(
                        [type_data], [year], [params], [dir_prelim], [dir_final]
                    )
                )
            )

    for val in all_params:
        download(val)

    ##########
    # Unzip
    ##########
    all_params = []
    for type_data in list_products:
        for year in range(start_year, end_year):
            for mon in range(1, 13):
                for day in range(1, monthrange(year, mon)[1] + 1):
                    all_params.extend(
                        list(
                            itertools.product(
                                [type_data],
                                [year],
                                [mon],
                                [day],
                                [params],
                                [dir_prelim],
                                [dir_final],
                            )
                        )
                    )

    if parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params), desc="unzipping CHIRPS") as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(unzip, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            unzip(val)

    ########################################
    # First process CHIRPS preliminary data
    # Next process CHIRPS final data (thereby overwriting the preliminary data in some places)
    ########################################
    for product in ["prelim", "final"]:
        all_params = []
        for year in range(start_year, end_year):
            sjd = 1
            jd_end = 366 if calendar.isleap(year) else 365

            for jd in range(sjd, jd_end):
                all_params.extend(
                    list(itertools.product([product], [year], [jd], [params]))
                )

        if parallel_process:
            with multiprocessing.Pool(
                int(multiprocessing.cpu_count() * fraction_cpus)
            ) as p:
                with tqdm(
                    total=len(all_params), desc=f"creating {product} CHIRPS global"
                ) as pbar:
                    for i, _ in tqdm(
                        enumerate(p.imap_unordered(to_global, all_params))
                    ):
                        pbar.update()
        else:
            for val in all_params:
                to_global(val)


if __name__ == "__main__":
    pass
