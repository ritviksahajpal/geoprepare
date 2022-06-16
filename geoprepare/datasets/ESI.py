# This scripts creates the global extent ESI tif files year and doy can be changed

import os
import pdb

import wget
import requests
import multiprocessing
import numpy as np
from tqdm import tqdm
from pathlib import Path
from osgeo.gdalnumeric import *

import always
import pygeoutil.util as util
import Code.base.constants as cc
import Code.preprocess.constants_preprocess as constants
import Code.base.log as log

logger = log.Logger(dir_log=cc.dir_tmp,
                    name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

syr = constants.START_YEAR
eyr = constants.END_YEAR

sjd = 8
ejd = 366

list_products = ['4wk']  #, '12wk']
base_url = 'https://gis1.servirglobal.net//data//esi//'


def download_ESI(all_params):
    """

    Args:
        all_params:

    Returns:

    """
    product, year = all_params[0], all_params[1]

    dir_download = constants.dir_download / 'esi' / product / str(year)
    util.make_dir_if_missing(dir_download)
    
    for jd in range(sjd, ejd, 7):
        # It is possible that the file is present as a tgz archive, in which case downloand and unzip it
        fl_download = f'DFPPM_{product.upper()}_{year}{str(jd).zfill(3)}.tif'

        # Download .tgz file if it has not been downloaded already or if we are updating data within the last year
        if not os.path.isfile(dir_download / fl_download):
            request = requests.head(base_url + str(product).upper()+ '//'+ str(year) + '//'+ fl_download)

            if request.status_code == 200:
                # logger.info('Downloading ' + fl_download)
                wget.download(base_url + str(product).upper()+ '//'+ str(year) + '//'+ fl_download, str(dir_download) + os.sep + fl_download)


def to_global(all_params):
    """

    Args:
        all_params:

    Returns:

    """
    from osgeo import osr

    product, year = all_params[0], all_params[1]

    inpath = Path(os.path.normpath(constants.dir_download / 'esi' / product / str(year)))
    outpath = Path(os.path.normpath(constants.dir_intermed / f'esi_{product}'))

    util.make_dir_if_missing(outpath)

    for jd in range(sjd, ejd, 7):
        format = 'GTiff'
        name_in = f'DFPPM_{product.upper()}_{year}{str(jd).zfill(3)}.tif'
        name_out = f'esi_dfppm_{product}_{year}{str(jd).zfill(3)}.tif'

        # check if output file does not exist but input file does
        if not os.path.exists(outpath / name_out) and os.path.isfile(inpath / name_in):
            logger.info(f'Creating {outpath} / {name_out}')

            ds = gdal.Open(str(inpath / name_in))

            b = ds.GetRasterBand(1)
            bArr = gdal.Band.ReadAsArray(b)
            outArr = np.empty([3600, 7200], dtype=float)
            outArr[3000:3600, :] = -9999.0
            outArr[0:3000, :] = bArr

            out = outpath / f'esi_dfppm_{product}_{year}{str(jd).zfill(3)}.tif'

            otype = gdal.GDT_Float32
            driver = gdal.GetDriverByName(format)
            dst_ds = driver.Create(str(out), 7200, 3600, 1, otype)
            outRasterSRS = osr.SpatialReference()
            outRasterSRS.ImportFromEPSG(4326)
            dst_ds.SetProjection(outRasterSRS.ExportToWkt())
            dst_ds.SetGeoTransform((-180, 0.05, 0, 90, 0, -0.05))
            dst_ds.GetRasterBand(1).WriteArray(outArr)
            dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)

            ds = None


if __name__ == '__main__':
    import itertools
    # Store git hash of current code
    logger.info('################ GIT HASH ################')
    logger.info(util.get_git_revision_hash())
    logger.info('################ GIT HASH ################')

    all_params = []
    for product in list_products:
        for year in range(syr, eyr):
            all_params.extend(list(itertools.product([product], [year])))

    # Download ESI data
    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(download_ESI, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            download_ESI(val)

    # Convert .tif to a global tif file
    all_params = []
    for product in list_products:
        for year in range(syr, eyr):
            all_params.extend(list(itertools.product([product], [year])))

    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(to_global, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            to_global(val)
