# README ###############################################################################################################
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
import rasterio
import itertools
import multiprocessing
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import always
import Code.base.log as log
import Code.base.constants as cc
import Code.preprocess.constants_preprocess as constants

logger = log.Logger(dir_log=cc.dir_tmp,
                    name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

syr = constants.START_YEAR
eyr = constants.END_YEAR
fill_value = constants.CHIRPS_Value

dir_final = constants.dir_download / 'chirps' / 'final'  # zipped tiff of CHIRPS final
dir_prelim = constants.dir_download / 'chirps' / 'prelim'  # unzipped tiff of CHIRPS prelim

list_products = ['prelim', 'final']


def download_CHIRPS(all_params):
    """
    https://data.chc.ucsb.edu/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/2020/
    https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/2020/
    Download CHIRPS preliminary and final files
    store prelim in: /cmongp1/GEOGLAM/Input/download/chirps/prelim
    store final in: /cmongp1/GEOGLAM/Input/download/chirps/final.
    :return:
    """
    from ftplib import FTP

    type_data, year = all_params[0], all_params[1]

    # prelim data should only be downloaded for current and previous year at most
    if type_data == 'prelim' and (always.redo_last_year and year < datetime.today().year - 1):
        return

    with FTP('ftp.chc.ucsb.edu', 'anonymous') as ftp:
        if type_data == 'prelim':
            try:
                ftp.cwd('/pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/' + str(year))
            except Exception as e:
                logger.error(type_data + ' ' + str(year) + ' ' + str(e))
                return
            dir_out = dir_prelim

        if type_data == 'final':
            ftp.cwd('/pub/org/chc/products/CHIRPS-2.0/global_daily/tifs/p05/' + str(year))
            dir_out = dir_final

        os.makedirs(dir_out, exist_ok=True)

        list_files = ftp.nlst()

        for fl in tqdm(list_files, desc='Downloading ' + type_data + ' CHIRPS'):
            if fl.find('chirps') > -1:
                local_filename = os.path.join(dir_out, fl)

                if os.path.isfile(dir_out / fl):
                    pass
                else:
                    with open(local_filename, 'wb') as lf:
                        ftp.retrbinary('RETR ' + fl, lf.write)


def unzip(all_params):
    """
    Store CHIRPS preliminary unzipped (/cmongp1/GEOGLAM/Input/intermed/chirps/prelim/unzipped)
    Store CHIRPS final scaled (/cmongp1/GEOGLAM/Input/intermed/chirps/final/scaled)
    Store CHIRPS prelim scaled (/cmongp1/GEOGLAM/Input/intermed/chirps/prelim/scaled)
    :param all_params:
    :return:
    """
    import gzip

    type_product, year, mon, day = all_params[0], all_params[1], all_params[2], all_params[3]

    if type_product == 'final':
        dir_unzipped = constants.dir_intermed / 'chirps' / type_product / 'unzipped'
        os.makedirs(dir_unzipped, exist_ok=True)

        # fl_zip refers to the downloaded zipped CHIRPS final file
        fl_zip = dir_final / Path(f'chirps-v2.0.{year}.{str(mon).zfill(2)}.{str(day).zfill(2)}.tif.gz')

    dir_integer = constants.dir_intermed / 'chirps' / type_product / 'scaled'
    os.makedirs(dir_integer, exist_ok=True)

    # change to JD
    day_of_year = datetime(year, mon, day).timetuple().tm_yday

    # fl_int is the unzipped and scaled file we create
    fl_int = dir_integer / Path(f'chirps_v2_{year}{str(day_of_year).zfill(3)}_scaled.tif')

    if os.path.exists(fl_int):
        pass
        # logger.info('File exists ' + fl_int)
    else:
        if type_product == 'final':
            fl_unzip = dir_unzipped / Path(f'chirps_v2.0.{year}{str(day_of_year).zfill(3)}_unzip.tif')

            if os.path.isfile(fl_zip):
                # logger.info('Unzipping ' + fl_zip + ' to ' + fl_unzip)
                with gzip.open(fl_zip, 'rb') as hndl_zip:
                    with open(fl_unzip, 'wb') as hndl_unzip:
                        hndl_unzip.write(hndl_zip.read())
        else:
            fl_unzip = dir_prelim / Path(f'chirps-v2.0.{year}.{str(mon).zfill(2)}.{str(day).zfill(2)}.tif')

        if os.path.isfile(fl_unzip):
            # logger.info('Converting to int ' + fl_unzip + ' to ' + fl_int)
            with rasterio.open(os.path.normpath(fl_unzip)) as dataset:
                profile = dataset.profile
                profile.update(dtype=rasterio.int32, count=1)
                data = dataset.read()

            arr = data * 100  # multiply by 100 to convert to integer
            arr[arr == -999900.0] = None  # Assigning null value

            with rasterio.open(fl_int, 'w', **profile) as dst:
                dst.write(arr.astype(rasterio.int32))


def to_global(all_params):
    """

    :param all_params:
    :return:
    """
    from osgeo import osr, gdal

    type_product, year, jd = all_params[0], all_params[1], all_params[2]

    dir_integer = constants.dir_intermed / 'chirps' / type_product / 'scaled'
    dir_out = constants.dir_intermed / 'chirps' / 'global'
    os.makedirs(dir_out, exist_ok=True)

    if os.path.isfile(dir_integer / Path('chirps_v2_' + str(year) + str(jd).zfill(3) + '_scaled.tif')):
        fl_out = dir_out / Path('chirps_v2.0.' + str(year) + str(jd).zfill(3) + '_global.tif')

        # Redo output file if it already exists, if it is this year or last
        if year < (datetime.today().year - 1) and os.path.isfile(fl_out):
            return

        # logger.info(type_product + ' global ' + dir_out + os.sep + 'chirps_v2.0.' + str(year) + str(jd).zfill(3) + '_global.tif')
        ds = gdal.Open(str(dir_integer) + os.sep + 'chirps_v2_' + str(year) + str(jd).zfill(3) + '_scaled.tif')

        b = ds.GetRasterBand(1)
        bArr = gdal.Band.ReadAsArray(b)
        outArr = np.empty([3600, 7200], dtype=int)
        outArr[0:800, :] = fill_value
        outArr[2800:3600, :] = fill_value
        outArr[800:2800, :] = bArr

        otype = gdal.GDT_Int32
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(str(fl_out), 7200, 3600, 1, otype)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
        dst_ds.GetRasterBand(1).WriteArray(outArr)

        ds = None
    else:
        pass
        # logger.info('Does not exist ' + dir_integer / 'chirps' + str(year) + str(jd).zfill(3) + '_scaled.tif')

if __name__ == '__main__':
    import calendar
    from calendar import monthrange

    ##########
    # DOWNLOAD
    ##########
    all_params = []
    for type_data in list_products:
        for year in range(syr, eyr):
            all_params.extend(list(itertools.product([type_data], [year])))

    for val in all_params:
        download_CHIRPS(val)

    ##########
    # Unzip
    ##########
    all_params = []
    for type_data in list_products:
        for year in range(syr, eyr):
            for mon in range(1, 13):
                for day in range(1, monthrange(year, mon)[1] + 1):
                    all_params.extend(list(itertools.product([type_data], [year], [mon], [day])))
    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params), desc='unzipping CHIRPS') as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(unzip, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            unzip(val)

    ########################################
    # First process CHIRPS preliminary data
    ########################################
    all_params = []
    for year in range(syr, eyr):
        sjd = constants.START_JD
        jd_end = constants.END_JD if calendar.isleap(year) else constants.END_JD2
        for jd in range(sjd, jd_end):
            all_params.extend(list(itertools.product(['prelim'], [year], [jd])))
    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params), desc='creating prelim CHIRPS global') as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(to_global, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            to_global(val)

    ##########################################################################################
    # Next process CHIRPS final data (thereby overwriting the preliminary data in some places)
    ##########################################################################################
    all_params = []
    for year in range(syr, eyr):
        sjd = constants.START_JD
        jd_end = constants.END_JD if calendar.isleap(year) else constants.END_JD2
        for jd in range(sjd, jd_end):
            all_params.extend(list(itertools.product(['final'], [year], [jd])))
    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params), desc='creating final CHIRPS global') as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(to_global, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            to_global(val)



