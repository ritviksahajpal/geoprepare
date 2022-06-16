import os
import pdb
import glob
import arrow as ar
import numpy as np
from tqdm import tqdm
from ftplib import FTP
from retrying import retry
from osgeo import osr, gdal
from pathlib import Path

import always
import pygeoutil.util as util
import Code.base.log as log
import Code.base.constants as cc
import Code.preprocess.constants_preprocess as constants

logger = log.Logger(dir_log=cc.dir_tmp, name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

syr = constants.START_YEAR
eyr = constants.END_YEAR
fill_value = constants.CHIRPS_Value


def get_forecast_file_name(list_files):
    """

    Args:
        list_files:

    Returns:

    """
    # Number of days to look back at if forecast file does not exist for current day
    MAX_DAYS = 15

    forecast_regex = 'data-mean_' + ar.utcnow().date().strftime('%Y%m%d')
    forecast_file = [s for s in list_files if forecast_regex in s]

    # If forecast file does not exist for current day then look at previous day (upto 14 days ago)
    if not len(forecast_file):
        logger.info('CHIRPS-GEFS data does not exist for ' + ar.utcnow().date().strftime('%Y-%m-%d'))
        for day in range(1, MAX_DAYS):
            forecast_regex = 'data-mean_' + ar.utcnow().shift(days=-day).date().strftime('%Y%m%d')
            forecast_file = [s for s in list_files if forecast_regex in s]

            # If data exists then break
            if len(forecast_file):
                logger.info('Getting CHIRPS-GEFS data from ' + ar.utcnow().shift(days=-day).date().strftime('%Y-%m-%d'))
                break

    return forecast_file


def delete_existing_files(dir_out):
    """

    Args:
        dir_out:

    Returns:

    """
    # Delete current file(s) in CHIRPS-GEFS directory
    filelist = glob.glob(os.path.join(dir_out, '*.tif'))

    for f in tqdm(filelist, desc='Deleting existing CHIRPS-GEFS file(s)'):
        # If today's file is already present, then do not delete
        forecast_regex = 'data-mean_' + ar.utcnow().date().strftime('%Y%m%d')
        if forecast_regex in f:
            continue

        # Else delete
        try:
            os.remove(f)
        except:
            logger.error('Unable to delete ' + f)


@retry(stop_max_attempt_number=3)
def download_CHIRPS_GEFS(dir_out):
    """
    Download CHIRPS-GEFS 15 day forecast files
    # ftp://chg-ftpout.geog.ucsb.edu/pub/org/chg/products/EWX/data/forecasts/CHIRPS-GEFS_precip/15day/precip_mean/
    /cmongp1/GEOGLAM/Input/download/chirps_gefs

    :return:
    """
    # Delete current file(s) in CHIRPS-GEFS directory
    delete_existing_files(dir_out)

    # Download last available file from CHIRPS-GEFS server
    with FTP('data.chc.ucsb.edu', 'anonymous') as ftp:
        try:
            ftp.cwd('/pub/org/chc/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/15day/precip_mean/')
        except Exception as e:
            # logger.error(type_data + ' ' + str(year) + ' ' + str(e))
            return

        list_files = ftp.nlst()

        # Extract based on current date and if that is not available, then yesterday
        forecast_file = get_forecast_file_name(list_files)

        if len(forecast_file):
            local_filename = os.path.join(dir_out, forecast_file[0])

            if os.path.isfile(dir_out / forecast_file[0]):
                pass
            else:
                logger.info('Downloading CHIRPS-GEFS file ' + local_filename)
                with open(local_filename, 'wb') as lf:
                    ftp.retrbinary('RETR ' + forecast_file[0], lf.write)


def to_global(dir_download):
    """

    :param all_params:
    :return:
    """
    # Get path of downloaded CHIRPS-GEFS file
    filelist = glob.glob(os.path.join(dir_download, '*.tif'))
    assert(len(filelist) == 1)

    forecast_file = filelist[0]

    dir_out = constants.dir_intermed / 'chirps_gefs'
    util.make_dir_if_missing(dir_out)
    delete_existing_files(dir_out)

    if not os.path.isfile(dir_out / os.path.basename(forecast_file)):
        fl_out = dir_out / os.path.basename(forecast_file)
        logger.info('Creating ' + str(fl_out))

        # logger.info(type_product + ' global ' + dir_out / 'chirps_v2.0.' + str(year) + str(jd).zfill(3) + '_global.tif')
        ds = gdal.Open(forecast_file)

        b = ds.GetRasterBand(1)
        bArr = gdal.Band.ReadAsArray(b)
        outArr = np.empty([3600, 7200], dtype=int)
        outArr[0:800, :] = fill_value
        outArr[2800:3600, :] = fill_value
        outArr[800:2800, :] = bArr * 100  # Multiply by 100 to match CHIRPS format

        otype = gdal.GDT_Int32
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(str(fl_out), 7200, 3600, 1, otype)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
        dst_ds.GetRasterBand(1).WriteArray(outArr)

        ds = None
        dst_ds = None


if __name__ == '__main__':
    # Store git hash of current code
    logger.info('################ GIT HASH ################')
    logger.info(util.get_git_revision_hash())
    logger.info('################ GIT HASH ################')

    ##########
    # DOWNLOAD
    ##########
    dir_download = constants.dir_download / 'chirps_gefs'
    util.make_dir_if_missing(dir_download)

    download_CHIRPS_GEFS(dir_download)
    to_global(dir_download)