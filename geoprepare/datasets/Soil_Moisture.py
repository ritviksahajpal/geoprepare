# README ##############################################################################################################
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
import sys
import ftplib

import always
import multiprocessing
import requests
import re
from bs4 import BeautifulSoup as soup
from tqdm import tqdm
from calendar import monthrange
from retrying import retry
from pathlib import Path

import pygeoutil.util as util
import Code.base.log as log
import Code.base.constants as cc
import Code.preprocess.constants_preprocess as constants

logger = log.Logger(dir_log=cc.dir_tmp, name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

syr = constants.START_YEAR
eyr = constants.END_YEAR
sjd = constants.START_JD
ejd = constants.END_JD

host_ftp = 'hrsl.ba.ars.usda.gov'

# Surface and Subsurface soil moisture:
#
# https://gimms.gsfc.nasa.gov/SMOS/jbolten/FAS/L03/
#
# Soil moisture profile:
#
# https://gimms.gsfc.nasa.gov/SMOS/jbolten/FAS/L04/
#
# Surface and Subsurface soil moisture anomaly:
#
# https://gimms.gsfc.nasa.gov/SMOS/jbolten/FAS/L05/

dir_download = constants.dir_download / 'soil_moisture_nasa_usda' / 'grib'
dir_tif = constants.dir_download / 'soil_moisture_nasa_usda' / 'tif'

util.make_dir_if_missing(dir_tif)
util.make_dir_if_missing(dir_download)

list_products = ['as2', 'as1']


@retry(stop_max_attempt_number=3)
def download_soil_moisture_out_of_date():
    try:
        ftp = ftplib.FTP(host_ftp, timeout=30)
        ftp.login('geoglam', 'geog8765')
    except Exception as e:
        raise ValueError(f'Cannot connect to {host_ftp}')

    dir_ftp = '/xport/ftp/pub/jbolten/FAS/L03/'
    ftp.cwd(dir_ftp)
    list_files = ftp.nlst()

    for year in tqdm(range(syr, eyr), desc='downloading soil moisture'):
        for fl in list_files:
            name_fl = str(fl)
            year_fl = name_fl[:4]
            is_surface = name_fl.split('.')[1]

            if int(year_fl) >= year:
                if os.path.isfile(dir_download / name_fl):
                    pass
                else:
                    dir_cur = os.getcwd()
                    os.chdir(str(dir_download))
                    ftp.retrbinary('RETR ' + name_fl, open(name_fl, 'wb').write)
                    os.chdir(dir_cur)


def download_soil_moisture():
    """

    :return:
    """
    url = 'https://gimms.gsfc.nasa.gov/SMOS/SMAP/L03/'

    data_html = soup(requests.get(url).text, 'lxml')

    # Find all grb2 files on page
    list_links = data_html.findAll(href=re.compile("/*.grb2$"))

    for idx, link in tqdm(enumerate(list_links), desc='download soil moisture data', total=len(list_links)):
        # If file has not been downloaded already, then download it
        name_fl = list_links[idx].get('href')

        if not os.path.isfile(str(dir_download) + os.sep + name_fl):
            download_link = url + data_html.findAll(href=re.compile("/*.grb2$"))[idx].get('href')

            request = requests.head(download_link)

            if request.status_code == 200:
                # logger.info('Downloading: ' + name_fl)
                r = requests.get(download_link, os.path.normpath(str(dir_download) + os.sep + name_fl))
                with open(str(dir_download) + os.sep + name_fl, 'wb') as f:
                    f.write(r.content)


def process_soil_moisture(all_params):
    """

    :param product:
    :param year:
    :param month:
    :param day:
    :return:
    """
    product, year, month, day = all_params[0], all_params[1], all_params[2], all_params[3]

    # change to JD
    day_of_year = datetime.date(year, month, day).timetuple().tm_yday

    dir_final = constants.dir_intermed / Path('soil_moisture_' + str(product))
    util.make_dir_if_missing(dir_final)

    fl_final = 'nasa_usda_soil_moisture_' + str(year) + '_' + str(day_of_year).zfill(3) + '_' + str(product) + '_global.tif'

    if not os.path.exists(dir_final / fl_final):
        file_search = glob.glob(str(dir_download) + os.sep + str(year) + str(month).zfill(2) + str(day).zfill(2) + '*' + str(product) + '.grb2')

        for fl in file_search:
            file = os.path.basename(fl)

            ras_input = os.path.normpath(dir_download / file)
            ras_interim = os.path.normpath(dir_tif / Path(file[:-4] + 'tif'))

            # Due to the nature of the grb2 files, a projection and extent first have to be forced onto the newly
            # created tifs using gdal_translate
            # logger.info('Forcing correct projection and extent: ' + ras_input + ' ' + ras_interim)
            # gdal_translate -a_srs EPSG:4326 -a_nodata -9999.0 -ot Float32 -of GTiff -a_ullr -179.9375, 89.9375, 180.0625, -60.0625
            # D:/Users/ritvik/projects/GEOGLAM/Input/crop_t20/20210121_20210123.as1.grb2
            # C:/Users/ritvik/AppData/Local/Temp/processing_weKHfd/2bfc4e98859f48d6b834b51629684b3b/OUTPUT.tif
            command = ['gdal_translate',
                       '-ot', 'Float32',
                       '-of', 'GTiff',
                       # '-a_ullr', '-179.9375','89.9375','180.0625','-60.0625',
                       # '-a_srs', 'EPSG:4326',
                       '-a_nodata', '9999.0',
                       os.path.normpath(ras_input),
                       ras_interim]
            subprocess.call(command)

            ras_final = dir_final / fl_final
            # logger.info('Changing to 0.05 degree global extent: ' + ras_interim + ' ' + ras_final)
            command = ['gdalwarp',
                       '-srcnodata', '-999.0',
                       '-dstnodata', '9999.0',
                       '-of', 'GTiff',
                       '-r', 'bilinear',
                       '-s_srs', 'EPSG:4326',
                       '-t_srs', 'EPSG:4326',
                       '-te', '-180', '-90', '180', '90',
                       '-ts', '7200', '3600',
                       ras_interim,
                       str(ras_final)]

            subprocess.call(command)
    else:
        pass
        # logger.info('file exists: ' + dir_final / fl_final)


if __name__ == '__main__':
    import itertools
    # Store git hash of current code
    logger.info('################ GIT HASH ################')
    logger.info(util.get_git_revision_hash())
    logger.info('################ GIT HASH ################')

    # download_soil_moisture_out_of_date()
    try:
        download_soil_moisture()
    except Exception as e:
        logger.error('Download of soil moisture data failed')

    all_params = []
    for product in list_products:
        for year in range(syr, eyr):
            for month in range(1, 13):
                for day in range(1, monthrange(year, month)[1] + 1):
                    all_params.extend(list(itertools.product([product], [year], [month], [day])))

    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(process_soil_moisture, all_params))):
                    pbar.update()
    else:
        pbar = tqdm(all_params, total=len(all_params))
        for val in pbar:
            pbar.set_description(str(val))
            process_soil_moisture(val)
            pbar.update()
