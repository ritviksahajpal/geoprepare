###############################################################################
# Ritvik Sahajpal, Joanne Hall
# ritvik@umd.edu
###############################################################################
import os
import itertools
import pdb

import pathlib
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

from geoprepare import utils

path_template = "template.nc"

profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": None,
    "width": 7200,
    "height": 3600,
    "count": 1,
    "crs": rasterio.crs.CRS({"init": "epsg:4326"}),
    "transform": Affine(0.05, 0.0, -180.0, 0.0, -0.05, 90.0),
    "tiled": False,
    "interleave": "band",
}


def download_nc(params, path_ftp, path_out, name_var):
    """

    :param path_ftp:
    :param path_out:
    :param name_var:
    :return:
    """
    os.makedirs(path_out / name_var, exist_ok=True)

    for year in tqdm(
        range(params.start_year, params.end_year + 1),
        desc=f"Downloading CPC {name_var}",
    ):
        path_nc = path_ftp + "/" + f"{name_var}.{year}.nc"
        path_out_yr = path_out / name_var / f"{name_var}.{year}.nc"

        # Download iff
        # 1. file does not exist
        # 2. file exists but is for previous year and it is not March yet (thereby waiting for 60 days)
        # 3. downloading present year data
        if (
            not os.path.isfile(path_out_yr)
            or (
                os.path.isfile(path_out_yr)
                and params.redo_last_year
                and (datetime.today().year - 1) == year
            )
            or (datetime.today().year == year)
        ):
            try:
                params.logger.info(
                    f"Downloading {name_var} from {path_nc} to {path_out_yr}"
                )
                if os.path.isfile(path_out_yr):
                    os.remove(path_out_yr)
                wget.download(path_nc, str(path_out_yr))
            except:
                params.logger.error(f"Download failed: {path_nc}")


def remap_like(original_nc, target_nc, name_var, index=0):
    """

    :param original_nc:
    :param target_nc:
    :param name_var:
    :param index:
    :return:
    """
    hndl_original = utils.convert_to_nc_hndl(original_nc)
    hndl_target = utils.convert_to_nc_hndl(target_nc)

    lat = hndl_original.variables["lat"].values
    lon = hndl_original.variables["lon"].values
    lon2d, lat2d = np.meshgrid(lon - 180, lat)
    orig_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    lat = hndl_target.variables["latitude"].values
    lon = hndl_target.variables["longitude"].values
    lon2d, lat2d = np.meshgrid(lon - 180, lat)
    targ_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    var = hndl_original.variables[name_var].values[index]
    remapped_var = pyresample.kd_tree.resample_nearest(
        orig_def, var, targ_def, radius_of_influence=5e5, fill_value=None
    )

    return remapped_var


def process_CPC(all_params):
    """

    :param var:
    :return:
    """
    params, var, year = all_params

    dir_output = params.dir_interim / f"cpc_{var}"
    os.makedirs(dir_output, exist_ok=True)

    dir_nc = params.dir_download / "cpc" / "original" / var

    nc_input = dir_nc / Path(f"{var}.{year}.nc")
    hndl_nc = xr.open_dataset(nc_input)

    for idx, doy in tqdm(
        enumerate(hndl_nc.variables["time"].values), desc=f"processing CPC {var} {year}"
    ):
        fl_out = f"cpc_{year}{str(pd.DatetimeIndex([doy]).dayofyear[0]).zfill(3)}_{var}_global.tif"

        if not os.path.isfile(dir_output / fl_out):
            arr = remap_like(
                nc_input,
                pathlib.Path(__file__).parent.resolve() / path_template,
                name_var=var,
                index=idx,
            )
            arr = np.roll(arr.data, int(arr.data.shape[1] / 2.0))

            utils.arr_to_tif(arr, dir_output / fl_out, profile)


def parallel_process_CPC(params):
    all_params = []

    for product in ["precip", "tmax", "tmin"]:
        for year in range(params.start_year, params.end_year + 1):
            all_params.extend(list(itertools.product([params], [product], [year])))

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(process_CPC, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            process_CPC(val)


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    download_nc(
        params,
        path_ftp=f"{params.data_dir}/cpc_global_precip",
        path_out=params.dir_download / "cpc" / "original",
        name_var="precip",
    )

    download_nc(
        params,
        path_ftp=f"{params.data_dir}/cpc_global_temp",
        path_out=params.dir_download / "cpc" / "original",
        name_var="tmax",
    )

    download_nc(
        params,
        path_ftp=f"{params.data_dir}/cpc_global_temp",
        path_out=params.dir_download / "cpc" / "original",
        name_var="tmin",
    )

    parallel_process_CPC(params)


if __name__ == "__main__":
    pass
