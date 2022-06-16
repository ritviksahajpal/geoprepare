import os
import pdb
import netCDF4
import itertools
import pyresample
import rasterio

import xarray as xr
import rioxarray as rix
import pandas as pd
import numpy as np
from tqdm import tqdm
from affine import Affine
from datetime import datetime
from pathlib import Path
import multiprocessing
import matplotlib.pyplot as plt

import always
import Code.preprocess.constants_preprocess as constants
import pygeoutil.util as util
import Code.base.log as log
import Code.base.constants as cc

logger = log.Logger(dir_log=cc.dir_tmp, name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

syr = 1982
eyr = constants.END_YEAR

template_fl = 'template.tif'


def remap_like(original_nc, target_nc, year, mon, name_var, index=0):
    """

    :param original_nc:
    :param target_nc:
    :param name_var:
    :param index:
    :return:
    """
    util.make_dir_if_missing(constants.dir_intermed / 'fldas' / 'prelim' / name_var)
    util.make_dir_if_missing(constants.dir_intermed / 'fldas' / 'final' / name_var)

    prelim_fl = constants.dir_intermed / 'fldas' / 'prelim' / name_var / f'{name_var}_{year}_{mon}.tif'
    final_fl = constants.dir_intermed / 'fldas' / 'final' / name_var / f'{name_var}_{year}_{mon}.tif'

    if not os.path.isfile(prelim_fl):
        # First create a tif file from netCDF file
        rds = xr.open_dataset(original_nc)[name_var]
        rds = rds.rename({'X': 'longitude', 'Y': 'latitude'})
        rds.rio.write_crs("epsg:4326", inplace=True)
        rds.rio.to_raster(prelim_fl)

        # Then rematch tif file to correct resolution
    if not os.path.isfile(final_fl):
        xds = xr.open_dataarray(prelim_fl)
        xds_match = xr.open_dataarray(template_fl)

        xds_repr_match = xds.rio.reproject_match(xds_match)
        xds_repr_match.rio.to_raster(final_fl)


def process_FLDAS(all_params):
    path_nc = all_params[0]

    list_vars = ['Evap_tavg', 'SoilMoi00_10cm_tavg', 'SoilMoi10_40cm_tavg', 'SoilMoi40_100cm_tavg', 'Tair_f_tavg']
    dir_nc = constants.dir_download / 'fldas'

    pbar = tqdm(path_nc)
    for fl in pbar:
        year = fl.split('_')[-1].split('.')[1][1:5]
        mon = fl.split('_')[-1].split('.')[1][-2:]

        for var in list_vars:
            pbar.set_description(f'FLDAS {year} {mon} {var}')
            dir_output = constants.dir_intermed / var
            util.make_dir_if_missing(dir_output)

            fl_out = f'fldas_{year}_{mon}_{var}.tif'

            nc_input = dir_nc / fl
            if not os.path.isfile(dir_output / fl_out):
                remap_like(nc_input, template_fl, year, mon, name_var=var, index=0)


def parallel_process_FLDAS():
    all_params = []

    dir_nc = constants.dir_download / 'fldas'
    nc_files = [f for f in os.listdir(dir_nc) if f.endswith('.nc')]

    all_params.extend(list(itertools.product([nc_files])))

    if False and constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(process_FLDAS, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            process_FLDAS(val)


if __name__ == '__main__':
    parallel_process_FLDAS()