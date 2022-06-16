# README ##############################################################################################################
# Data Preprocessing
# The data is already provided at a 0.05 degree resolution, global extent to match the crop masks.
# In order to save space on the server, the files have been compressed so a script is written to uncompress the files.
#
# Data Extraction
# Values range between 0 - 250. Fill values are between 251 - 255. Fill values are masked out in the extraction code.
# Scaled by (NDVI * 200) + 50
# True 0 NDVI values are in fact '50' in the data layers
# '0' in the data layers are equivalent of missing data (i.e. over oceans etc)
#######################################################################################################################
import os
import pdb
import always
import requests
import re
from bs4 import BeautifulSoup as soup
import urllib.request
import subprocess
import multiprocessing
from tqdm import tqdm
import pygeoutil.util as util

import Code.preprocess.constants_preprocess as constants
import Code.base.log as log
logger = log.Logger(name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

base_url = "http://pekko.geog.umd.edu/users/bbarker/mod09.ndvi.global_0.05_degree.v1/"

syr = constants.START_YEAR
eyr = constants.END_YEAR
sjd = constants.START_JD
ejd = constants.END_JD


def download_NDVI():
    """
    pekko.geog.umd.edu/users/bbarker/mod09.ndvi.global_0.05_degree.v1/mod09.ndvi.global_0.05_degree.2018.001.c6.v1.tif
    :return:
    """
    dir_download = constants.dir_download + os.sep + 'ndvi'
    util.make_dir_if_missing(dir_download)

    data_html = soup(requests.get(base_url).text, 'lxml')

    # Find all grib files on page
    list_links = data_html.findAll(href=re.compile("/*.tif$"))

    for idx, link in enumerate(list_links):
        # If file has not been downloaded already, then download it
        name_fl = list_links[idx].get('href')

        if not os.path.isfile(dir_download + os.sep + name_fl):
            download_link = base_url + name_fl

            request = requests.head(download_link)

            if request.status_code == 200:
                path_download = os.path.normpath(dir_download + os.sep + name_fl)
                logger.info('downloading ' + path_download)

                r = requests.get(download_link, path_download)
                with open(path_download, 'wb') as f:
                    f.write(r.content)


def process_NDVI(all_params):
    """

    :param all_params:
    :return:
    """
    dir_download = constants.dir_download + os.sep + 'ndvi'
    dir_out = constants.dir_intermed + os.sep + 'ndvi'
    util.make_dir_if_missing(dir_out)

    year, jd = all_params[0], all_params[1]

    name_file = 'mod09.ndvi.global_0.05_degree.' + str(year) + '.' + str(jd).zfill(3) + '.c6.v1.tif'

    if os.path.isfile(dir_download + os.sep + name_file):
        if not os.path.isfile(dir_out + os.sep + name_file):
            path_out = dir_out + os.sep + name_file
            fl = dir_download + os.sep + name_file

            # logger.info('Uncompressing ' + fl + ' to ' + path_out)
            tiff_command = ['gdal_translate', '-ot', 'Byte', '-of', 'GTiff', '-co', 'COMPRESS=LZW', fl, path_out]
            subprocess.call(tiff_command)
        else:
            pass
            # logger.info('File exists: ' + dir_out + os.sep + name_file)


if __name__ == '__main__':
    import itertools

    # Store git hash of current code
    logger.info('################ GIT HASH ################')
    logger.info(util.get_git_revision_hash())
    logger.info('################ GIT HASH ################')

    download_NDVI()

    all_params = []
    for year in range(syr, eyr):
        for jd in range(sjd, ejd, 8):
            all_params.extend(list(itertools.product([year], [jd])))

    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(process_NDVI, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            process_NDVI(val)
