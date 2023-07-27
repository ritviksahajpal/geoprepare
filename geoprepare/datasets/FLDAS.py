###############################################################################
# Ritvik Sahajpal
# ritvik@umd.edu
###############################################################################
import os
import pdb
import itertools

import xarray as xr
from tqdm import tqdm
import multiprocessing

template_fl = "template.tif"


def remap_like(params, original_nc, year, mon, name_var):
    """

    Args:
        original_nc ():
        target_nc ():
        year ():
        mon ():
        name_var ():
        index ():

    Returns:

    """
    os.makedirs(params.dir_interim / "fldas" / "prelim" / name_var, exist_ok=True)
    os.makedirs(params.dir_interim / "fldas" / "final" / name_var, exist_ok=True)

    prelim_fl = (
        params.dir_interim
        / "fldas"
        / "prelim"
        / name_var
        / f"{name_var}_{year}_{mon}.tif"
    )
    final_fl = (
        params.dir_interim
        / "fldas"
        / "final"
        / name_var
        / f"{name_var}_{year}_{mon}.tif"
    )

    if not os.path.isfile(prelim_fl):
        # First create a tif file from netCDF file
        rds = xr.open_dataset(original_nc)[name_var]
        rds = rds.rename({"X": "longitude", "Y": "latitude"})
        rds.rio.write_crs("epsg:4326", inplace=True)
        rds.rio.to_raster(prelim_fl)

    # Then rematch tif file to correct resolution
    if not os.path.isfile(final_fl):
        xds = xr.open_dataarray(prelim_fl)
        xds_match = xr.open_dataarray(template_fl)

        xds_repr_match = xds.rio.reproject_match(xds_match)
        xds_repr_match.rio.to_raster(final_fl)


def process_FLDAS(all_params):
    params, path_nc = all_params

    list_vars = [
        "Evap_tavg",
        "SoilMoi00_10cm_tavg",
        "SoilMoi10_40cm_tavg",
        "SoilMoi40_100cm_tavg",
        "Tair_f_tavg",
    ]
    dir_nc = params.dir_download / "fldas"

    pbar = tqdm(path_nc)
    for fl in pbar:
        year = fl.split("_")[-1].split(".")[1][1:5]
        mon = fl.split("_")[-1].split(".")[1][-2:]

        for var in list_vars:
            pbar.set_description(f"FLDAS {year} {mon} {var}")
            dir_output = params.dir_interim / var
            os.makedirs(dir_output, exist_ok=True)

            fl_out = f"fldas_{year}_{mon}_{var}.tif"

            nc_input = dir_nc / fl
            if not os.path.isfile(dir_output / fl_out):
                remap_like(params, nc_input, year, mon, name_var=var)


def run(params):
    all_params = []

    dir_nc = params.dir_download / "fldas"
    nc_files = [f for f in os.listdir(dir_nc) if f.endswith(".nc")]

    all_params.extend(list(itertools.product([params], [nc_files])))

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cups)
        ) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(
                    enumerate(p.imap_unordered(process_FLDAS, all_params))
                ):
                    pbar.update()
    else:
        for val in all_params:
            process_FLDAS(val)


if __name__ == "__main__":
    pass
