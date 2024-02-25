###############################################################################
# Ritvik Sahajpal
# ritvik@umd.edu
###############################################################################
import os
import pdb
import itertools
import pathlib
import cdsapi
import urllib3
import pyresample
import rasterio
import multiprocessing
import zipfile
import pandas as pd
import numpy as np
from datetime import timedelta
from calendar import monthrange
from tqdm import tqdm
from affine import Affine
from datetime import datetime
from pathlib import Path

from geoprepare import utils

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

variable_names = {
    "Temperature_Air_2m_Mean_24h": ["2m_temperature", "24_hour_mean"],
    "Temperature_Air_2m_Mean_Day_Time": ["2m_temperature", "day_time_mean"],
    "Temperature_Air_2m_Mean_Night_Time": ["2m_temperature", "night_time_mean"],
    "Dew_Point_Temperature_2m_Mean": ["2m_dewpoint_temperature", "24_hour_mean"],
    "Temperature_Air_2m_Max_24h": ["2m_temperature", "24_hour_maximum"],
    "Temperature_Air_2m_Min_24h": ["2m_temperature", "24_hour_minimum"],
    "Temperature_Air_2m_Max_Day_Time": ["2m_temperature", "day_time_maximum"],
    "Temperature_Air_2m_Min_Night_Time": ["2m_temperature", "night_time_minimum"],
    "Precipitation_Flux": ["precipitation_flux", None],
    "Snow_Thickness_Mean": ["snow_thickness", "24_hour_mean"],
    "Solar_Radiation_Flux": ["solar_radiation_flux", None],
    "Vapour_Pressure_Mean": ["vapour_pressure", "24_hour_mean"],
}


def get_list_days(year, month):
    """
    Returns a list of days in a given month
    :param year: the year
    :param month: the month
    :return: a list of days in the given month
    """
    days = monthrange(year, month)[1]

    return [day for day in range(1, days + 1)]


def create_target_fname(
    meteo_variable_full_name, sday, agera5_dir, stat="final", v="1.1"
):
    """
    Creates the AgERA5 variable filename for given variable and day
    https://github.com/ajwdewit/agera5tools/blob/main/agera5tools/util.py
    :param meteo_variable_full_name: the full name of the meteo variable
    :param day: the date of the file
    :param agera5_dir: the path to the AgERA5 dataset
    :return: the full path to the target filename
    """
    name_with_dashes = meteo_variable_full_name.replace("_", "-")

    nc_fname = (
        Path(agera5_dir)
        / f"{name_with_dashes}_C3S-glob-agric_AgERA5_{sday}_{stat}-v{v}.nc"
    )

    return nc_fname


def get_date_from_fname(path_file):
    # Get name of file from file path fname
    fname = os.path.basename(path_file)

    # Get date from filename fname and convert to date object
    date_string = fname.split("_")[-2]

    # convert date string to date object
    date = datetime.strptime(date_string, "%Y%m%d")

    return date


def remap_like(original_nc, target_nc, name_var, index=0):
    """

    Args:
        original_nc ():
        target_nc ():
        name_var ():
        index ():

    Returns:

    """
    hndl_original = utils.convert_to_nc_hndl(original_nc)
    hndl_target = utils.convert_to_nc_hndl(target_nc)

    lat = hndl_original.variables["lat"].values
    lon = hndl_original.variables["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    orig_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    lat = hndl_target.variables["latitude"].values
    lon = hndl_target.variables["longitude"].values
    lon2d, lat2d = np.meshgrid(lon - 180, lat)
    targ_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    var = hndl_original.variables[name_var].values[index]
    remapped_var = pyresample.kd_tree.resample_nearest(
        orig_def, var, targ_def, radius_of_influence=5000, fill_value=None
    )

    return remapped_var


def process_agERA5(all_params):
    """

    Args:
        all_params ():

    Returns:

    """
    var, nc_input, dir_output = all_params

    date = get_date_from_fname(nc_input)
    year = date.year

    fl_out = f"agera5_{year}{str(pd.DatetimeIndex([date]).dayofyear[0]).zfill(3)}_{var}_global.tif"

    if not os.path.isfile(dir_output / fl_out):
        arr = remap_like(
            nc_input,
            pathlib.Path(__file__).parent.resolve() / path_template,
            name_var=var,
        )
        # arr = np.roll(arr.data, int(arr.data.shape[1] / 2.))

        utils.arr_to_tif(arr, dir_output / fl_out, profile)


def parallel_process_agERA5(params):
    """

    Args:
        params ():

    Returns:

    """
    all_params = []

    for var in list(variable_names.keys()):
        dir_output = params.dir_interim / "agera5" / "tif" / var
        os.makedirs(dir_output, exist_ok=True)

        dir_nc = params.dir_interim / "agera5" / "nc" / var

        nc_files = dir_nc.glob("*.nc")
        for nc_input in nc_files:
            all_params.extend(list(itertools.product([var], [nc_input], [dir_output])))

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(
                    enumerate(p.imap_unordered(process_agERA5, all_params))
                ):
                    pbar.update()
    else:
        pbar = tqdm(all_params)
        for val in pbar:
            pbar.set_description(f"Processing {val[0]} {val[1]} {val[2]}")
            pbar.update()

            process_agERA5(val)


def download_nc(inputs, version="1.1"):
    """

    Args:
        inputs ():
        version ():

    Returns:

    """
    c, params, path_download, path_nc, varname, year, mon = inputs
    os.makedirs(path_download / varname, exist_ok=True)
    os.makedirs(path_nc / varname, exist_ok=True)

    days = get_list_days(year, mon)
    statistic = variable_names.get(varname)[1]
    variable = variable_names.get(varname)[0]

    for day in days:
        # create output filename for netcdf file
        sday = datetime(year, mon, day).strftime("%Y%m%d")

        # AgERA5 does not have data for the most recent month, so we need to skip them
        # by checking if sday is more recent than 30 days in the past
        if sday > (datetime.now() - timedelta(days=10)).strftime("%Y%m%d"):
            break

        fname = create_target_fname(varname, sday, path_nc / varname, v="1.1")
        fzip = path_download / varname / f"{varname}_{sday}_{statistic}-v{version}.zip"

        # Check if netCDF file already exists, if not then download it
        if not os.path.exists(fname):
            try:
                if statistic:
                    c.retrieve(
                        "sis-agrometeorological-indicators",
                        {
                            'version': '1_1',
                            "variable": variable,
                            "statistic": statistic,
                            "year": str(year),
                            "month": str(mon).zfill(2),
                            "day": str(day).zfill(2),
                            "format": "zip",
                        },
                        str(fzip),
                    )
                else:
                    c.retrieve(
                        "sis-agrometeorological-indicators",
                        {
                            'version': '1_1',
                            "variable": variable,
                            "year": str(year),
                            "month": str(mon).zfill(2),
                            "day": str(day).zfill(2),
                            "format": "zip",
                        },
                        str(fzip),
                    )
            except Exception as e:
                params.logger.error(f"Could not download {fname} {e}")

            # Unzip file to get the netCDF file
            if os.path.isfile(fzip):
                with zipfile.ZipFile(fzip, "r") as zip_ref:
                    zip_ref.extractall(path_nc / varname)


def download_parallel_nc(c, params, path_download, path_nc, variable):
    """

    Args:
        params ():
        path_download ():
        path_nc ():
        variable ():

    Returns:

    """
    all_params = []
    for year in range(params.start_year, params.end_year + 1):
        for mon in range(1, 13):
            all_params.extend(
                list(
                    itertools.product(
                        [c],
                        [params],
                        [path_download],
                        [path_nc],
                        [variable],
                        [year],
                        [mon],
                    )
                )
            )

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(download_nc, all_params))):
                    pbar.set_description(
                        f"Downloading AgERA5 {variable} year:{all_params[i][5]} month:{all_params[i][6]}"
                    )
                    pbar.update()
    else:
        for param in tqdm(
            all_params, desc=f"Downloading AgERA5 {variable} year:{year} month:{mon}"
        ):
            download_nc(param)


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    path_download = params.dir_download / "agera5"
    path_nc = params.dir_interim / "agera5" / "nc"

    try:
        c = cdsapi.Client()
    except Exception as e:
        params.logger.error(f"Cannot connect to CDSAPI: {e}")
        exit(1)

    for variable in variable_names.keys():
        if variable in params.variables:
            download_parallel_nc(c, params, path_download, path_nc, variable)

    parallel_process_agERA5(params)


if __name__ == "__main__":
    pass
