###############################################################################
# Ritvik Sahajpal, Joanne Hall
# ritvik@umd.edu
###############################################################################
import os
import pdb

import wget
import requests
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from osgeo.gdalnumeric import *
import numpy as np
from tqdm import tqdm

start_jd = 8
end_jd = 366


def download_ESI(all_params):
    """

    Args:
        all_params:

    Returns:

    """
    params, product, year = all_params

    dir_download = params.dir_download / "esi" / product / str(year)
    os.makedirs(dir_download, exist_ok=True)

    pbar = tqdm(range(start_jd, end_jd, 7))
    for jd in pbar:
        pbar.set_description(f"Downloading {product} {year} {jd}")
        pbar.update()

        # It is possible that the file is present as a tgz archive, in which case downloand and unzip it
        fl_download = f"DFPPM_{product.upper()}_{year}{str(jd).zfill(3)}.tif"

        # Download .tgz file if it has not been downloaded already or if we are updating data within the last year
        if not os.path.isfile(dir_download / fl_download):
            request = requests.head(
                f"{params.data_dir}{product.upper()}//{year}//{fl_download}"
            )

            if request.status_code == 200:
                # params.logger.info('Downloading ' + fl_download)
                wget.download(
                    f"{params.data_dir}{product.upper()}//{year}//{fl_download}",
                    f"{dir_download}/{fl_download}",
                )


def to_global(all_params):
    """

    Args:
        all_params:

    Returns:

    """
    from osgeo import osr

    params, product, year = all_params

    inpath = Path(os.path.normpath(params.dir_download / "esi" / product / str(year)))
    outpath = Path(os.path.normpath(params.dir_interim / f"esi_{product}"))

    os.makedirs(outpath, exist_ok=True)

    for jd in range(start_jd, end_jd, 7):
        format = "GTiff"
        name_in = f"DFPPM_{product.upper()}_{year}{str(jd).zfill(3)}.tif"
        name_out = f"esi_dfppm_{product}_{year}{str(jd).zfill(3)}.tif"

        # check if output file does not exist but input file does
        if not os.path.exists(outpath / name_out) and os.path.isfile(inpath / name_in):
            params.logger.info(f"Creating {outpath} / {name_out}")

            ds = gdal.Open(str(inpath / name_in))

            b = ds.GetRasterBand(1)
            bArr = gdal.Band.ReadAsArray(b)
            outArr = np.empty([3600, 7200], dtype=float)
            outArr[3000:3600, :] = -9999.0
            outArr[0:3000, :] = bArr

            out = outpath / f"esi_dfppm_{product}_{year}{str(jd).zfill(3)}.tif"

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


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    import itertools

    all_params = []
    for product in params.list_products:
        for year in range(params.start_year, params.end_year + 1):
            all_params.extend(list(itertools.product([params], [product], [year])))

    # Download ESI data
    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params), desc="download ESI") as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(download_ESI, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            download_ESI(val)

    # Convert .tif to a global tif file
    all_params = []
    for product in params.list_products:
        for year in range(params.start_year, params.end_year + 1):
            all_params.extend(list(itertools.product([params], [product], [year])))

    if params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params), desc="process ESI") as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(to_global, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            to_global(val)


if __name__ == "__main__":
    pass
