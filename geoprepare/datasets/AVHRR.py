###############################################################################
# Ritvik Sahajpal
# ritvik@umd.edu
###############################################################################
import os
import pdb

import glob
import datetime
import requests
import rasterio
import itertools

import multiprocessing
import numpy as np
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
from affine import Affine

from geoprepare import utils

path_template = "template.nc"

profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": -9999.0,
    "width": 7200,
    "height": 3600,
    "count": 1,
    "crs": rasterio.crs.CRS({"init": "epsg:4326"}),
    "transform": Affine(0.05, 0.0, -180.0, 0.0, -0.05, 90.0),
    "tiled": False,
    "interleave": "band",
}


def download_AVHRR(all_params):
    """

    Args:
        all_params ():

    Returns:

    """
    params, year = all_params

    folder_location = params.dir_download / "avhrr_v5" / str(year)
    os.makedirs(folder_location, exist_ok=True)

    response = requests.get(f"{params.data_dir}/{year}")
    soup = BeautifulSoup(response.text, "html.parser")

    for link in tqdm(soup.select("a[href$='.nc']"), desc=f"AVHRR {year}"):
        # Name the pdf files using the last portion of each link which are unique in this case
        filename = os.path.join(folder_location, link["href"].split("/")[-1])

        if not os.path.isfile(filename):
            with open(filename, "wb") as f:
                f.write(
                    requests.get(
                        urljoin(params.data_dir + "/" + str(year) + "/", link["href"])
                    ).content
                )


def process_AVHRR(all_params):
    """

    :param var:
    :return:
    """
    params, year = all_params
    var = "NDVI"

    dir_output = params.dir_interim / "avhrr_v5" / str(year)
    os.makedirs(dir_output, exist_ok=True)

    dir_nc = params.dir_download / "avhrr_v5" / str(year)

    nc_files = [f for f in os.listdir(dir_nc) if f.endswith(".nc")]

    for fl in tqdm(nc_files, desc=f"netCDF to tif {year}", leave=False):
        mon = fl.split("_")[-2][4:6]
        dom = fl.split("_")[-2][6:]
        dt = datetime.datetime.strptime(f"{year}-{mon}-{dom}", "%Y-%m-%d")
        doy = dt.timetuple().tm_yday

        fl_out = f"avhrr_v5_{year}_{str(doy).zfill(3)}.tif"

        hndl_nc = utils.convert_to_nc_hndl(dir_nc / fl)
        if not os.path.isfile(dir_output / fl_out):
            arr = hndl_nc.variables[var].values[0]
            utils.arr_to_tif(arr, dir_output / fl_out, profile)


def mvc(file_list, path_out=None, use_temporary=False):
    """
    Maximum value composite
    Args:
        file_list:
        path_out:
        use_temporary:

    Returns:

    """
    if use_temporary:
        import tempfile

        path_out = tempfile.NamedTemporaryFile(suffix=".tif").name

    # Read all data as a list of numpy arrays
    ls_arrays = [rasterio.open(x).read(1) for x in file_list]

    # Perform MVC
    arr = np.nanmax(ls_arrays, axis=0)

    # Get metadata from one of the input files
    with rasterio.open(file_list[0]) as src:
        meta = src.meta

    # Write output file
    with rasterio.open(path_out, "w", **meta) as dst:
        dst.write(arr, 1)

    if use_temporary:
        return path_out


def create_composite(params, suffix_out_dir, chunk_size=10):
    """

    Args:
        suffix_out_dir:
        chunk_size:

    Returns:

    """
    dir_output = params.dir_interim / "avhrr_v5" / "composite" / suffix_out_dir
    dir_input = params.dir_interim / "avhrr_v5"
    os.makedirs(dir_output, exist_ok=True)

    tif_files = glob.glob(f"{dir_input}/**/*.tif", recursive=True)
    tif_files = sorted(
        tif_files, key=lambda x: [x.split("_")[-2], x.split("_")[-1][:-4]]
    )

    # exclude elements from tif_files which have composite
    tif_files = [x for x in tif_files if "composite" not in x]

    # Divide into chunks of approx size chunk_size
    chunks = np.array_split(np.array(tif_files), len(tif_files) // chunk_size)

    for chunk in tqdm(chunks, desc="MVC"):
        year = chunk[0].split("_")[-2]
        first_day = chunk[0].split("_")[-1][:-4]

        path_composite = (
            dir_output / f"composite_{chunk_size}_avhrr_v5_{year}_{first_day}.tif"
        )

        if not os.path.isfile(path_composite):
            mvc(chunk, path_composite)


def parallel_process_AVHRR(params):
    all_params = []

    for year in range(params.start_year, params.end_year + 1):
        all_params.extend(list(itertools.product([params], [year])))

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(
                    enumerate(p.imap_unordered(process_AVHRR, all_params))
                ):
                    pbar.update()
    else:
        for val in tqdm(all_params, desc="process AVHRR"):
            process_AVHRR(val)


def run(params):
    all_params = []
    for year in range(params.start_year, params.end_year + 1):
        all_params.extend(list(itertools.product([params], [year])))

    # Download AVHRR data
    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            for i, _ in enumerate(p.imap_unordered(download_AVHRR, all_params)):
                pass
    else:
        for val in all_params:
            download_AVHRR(val)

    # Process AVHRR files
    parallel_process_AVHRR(params)

    create_composite(params, suffix_out_dir="dekadal", chunk_size=10)
    create_composite(params, suffix_out_dir="monthly", chunk_size=30)


if __name__ == "__main__":
    pass
