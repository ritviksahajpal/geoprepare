import os
import pdb
import netCDF4
import itertools
import pyresample
import rasterio
import wget
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from affine import Affine
from datetime import datetime
from pathlib import Path
import multiprocessing

import always
import Code.preprocess.constants_preprocess as constants
import pygeoutil.util as util
import Code.base.log as log
import Code.base.constants as cc

logger = log.Logger(dir_log=cc.dir_tmp, name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

syr = constants.START_YEAR
eyr = constants.END_YEAR

path_template = 'template.nc'

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': 7200, 'height': 3600, 'count': 1,
           'crs': rasterio.crs.CRS({'init': 'epsg:4326'}), 'transform': Affine(0.05, 0.0, -180.0, 0.0, -0.05, 90.0),
           'tiled': False, 'interleave': 'band'}


def download_nc(path_ftp, path_out, name_var):
    """

    :param path_ftp:
    :param path_out:
    :param name_var:
    :return:
    """
    util.make_dir_if_missing(path_out / name_var)

    for year in tqdm(range(syr, eyr), desc=f'Downloading CPC {name_var}'):
        path_nc = path_ftp + '/' + f'{name_var}.{year}.nc'
        path_out_yr = path_out / name_var / f'{name_var}.{year}.nc'

        # Download iff
        # 1. file does not exist
        # 2. file exists but is for previous year and it is not March yet (thereby waiting for 60 days)
        # 3. downloading present year data
        if not os.path.isfile(path_out_yr) or \
                (os.path.isfile(path_out_yr) and always.redo_last_year and (datetime.today().year - 1) == year) or \
                (datetime.today().year == year):
            try:
                # logger.info('Downloading ' + name_var + ' from ' + path_nc + ' to ' + path_out_yr)
                if os.path.isfile(path_out_yr):
                    os.remove(path_out_yr)
                wget.download(path_nc, str(path_out_yr))
            except:
                logger.error(f'Download failed: {path_nc}')


def convert_to_nc_hndl(path_nc):
    """

    :param path_nc:
    :return:
    """
    hndl_nc = path_nc
    if not isinstance(path_nc, np.ma.MaskedArray):
        _, ext = os.path.splitext(path_nc)

    if ext in ['.nc', '.nc4']:
        hndl_nc = xr.open_dataset(path_nc)

    return hndl_nc


def remap_like(original_nc, target_nc, name_var, index=0):
    """

    :param original_nc:
    :param target_nc:
    :param name_var:
    :param index:
    :return:
    """
    hndl_original = convert_to_nc_hndl(original_nc)
    hndl_target = convert_to_nc_hndl(target_nc)

    lat = hndl_original.variables['lat'].values
    lon = hndl_original.variables['lon'].values
    lon2d, lat2d = np.meshgrid(lon - 180, lat)
    orig_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    lat = hndl_target.variables['latitude'].values
    lon = hndl_target.variables['longitude'].values
    lon2d, lat2d = np.meshgrid(lon - 180, lat)
    targ_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    var = hndl_original.variables[name_var].values[index]
    remapped_var = pyresample.kd_tree.resample_nearest(orig_def, var, targ_def, radius_of_influence=5e5, fill_value=None)

    return remapped_var


def arr_to_tif(arr, path_tif, profile):
    """

    :param arr:
    :param path_tif:
    :param profile:
    :return:
    """
    with rasterio.open(path_tif, 'w', **profile) as dst:
        dst.write(arr, 1)


def process_CPC(all_params):
    """

    :param var:
    :return:
    """
    var, year = all_params[0], all_params[1]

    dir_output = constants.dir_intermed / f'cpc_{var}'
    util.make_dir_if_missing(dir_output)

    dir_nc = constants.dir_download / 'cpc' / 'original' / var

    nc_input = dir_nc / Path(var + '.' + str(year) + '.nc')
    hndl_nc = util.open_or_die(nc_input, use_xarray=True)

    for idx, doy in tqdm(enumerate(hndl_nc.variables['time'].values), desc=f'processing CPC {var} {year}'):
        fl_out = f'cpc_{year}{str(pd.DatetimeIndex([doy]).dayofyear[0]).zfill(3)}_{var}_global.tif'

        if not os.path.isfile(dir_output / fl_out):
            arr = remap_like(nc_input, path_template, name_var=var, index=idx)
            arr = np.roll(arr.data, int(arr.data.shape[1]/2.))

            arr_to_tif(arr, dir_output / fl_out, profile)


def parallel_process_CPC():
    all_params = []

    for product in ['precip', 'tmax', 'tmin']:
        for year in range(syr, eyr):
            all_params.extend(list(itertools.product([product], [year])))

    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(process_CPC, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            process_CPC(val)


if __name__ == '__main__':
    # Store git hash of current code
    logger.info('################ GIT HASH ################')
    logger.info(util.get_git_revision_hash())
    logger.info('################ GIT HASH ################')

    download_nc(path_ftp='ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_precip',
                path_out=constants.dir_download / 'cpc' / 'original',
                name_var='precip')

    download_nc(path_ftp='ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_temp',
                path_out=constants.dir_download / 'cpc' / 'original',
                name_var='tmax')

    download_nc(path_ftp='ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_temp',
                path_out=constants.dir_download / 'cpc' / 'original',
                name_var='tmin')

    parallel_process_CPC()
