######################################################################################################################
# Ritvik Sahajpal, Jie Zhang
# ritvik@umd.edu
# Data Preprocessing
# Global Daily MODIS Land Surface Temperature (LST) data at 0.05Deg (MOD11C1.006)
# The data is already provided at a 0.05 degree resolution, global extent to match the crop masks.
# For each day, pixels with poor LST quality were masked out based on the quality assurance (QA) layer
# (Note: The two least significant bits "00" indicate LST produced with good quality).
# Daily LST data with good quality were output into daily tiff file.
# Values range between 7500-65535. Fill values are 0. Fill values are masked out in the Weighted Average Extraction code.
# The value is scaled Kelvin temperature and can be converted to Celsius temperature by LST=DN*0.02-273.15(°C).
#######################################################################################################################
import os
import glob
import pdb

import pathlib
import numpy as np
import multiprocessing
from tqdm import tqdm
from osgeo import gdal

import pymodis

start_jd = 1
end_jd = 367


def download_LST(params):
    # destination folder
    dir_download = params.dir_download / "modis_lst"
    os.makedirs(dir_download, exist_ok=True)

    modisDown = pymodis.downmodis.downModis(
        destinationFolder=dir_download,
        url="https://e4ftl01.cr.usgs.gov",
        tiles=None,
        path="MOLT",
        product="MOD11C1.006",
        delta=params.num_update_days,
    )
    modisDown.connect()

    pbar = tqdm(range(0, params.num_update_days, 1))
    for n in pbar:
        day = modisDown.getListDays()[n]
        pbar.set_description(f"Download MODIS LST for {day}")

        # get the list of data files for specific day
        list_new_file = modisDown.getFilesList(day)

        # check whether the data exists and return only the list of previously non-downloaded data files
        list_files_down = modisDown.checkDataExist(list_new_file, move=False)

        # download the previously non-downloaded data file
        if modisDown.nconnection <= 20:
            # maximum number of attempts to connect to the HTTP server before failing
            modisDown.dayDownload(day, list_files_down)
        else:
            params.logger.error("A problem with the connection occurred")


def qa_extraction(qa_infile):
    """
    Unpack qa data layer from a HDF5 container
    Args:
        qa_infile ():

    Returns:

    """

    # open the dataset
    qa_hdf_ds = gdal.Open(qa_infile, gdal.GA_ReadOnly)
    qa_ds = gdal.Open(qa_hdf_ds.GetSubDatasets()[1][0], gdal.GA_ReadOnly)

    # read into numpy array
    qa_array = qa_ds.ReadAsArray().astype(np.uint8)

    # read qa bit info, get the 2 least significant bits for quality check ('00' means good LST quality)
    mask = (qa_array & 3) == 0

    # mask pixels without good LST quality as NaN
    mask[mask == 0.0] = np.NaN

    return mask


def hdf_subdataset_extraction(hdf_file):
    """
    Retrieve the LST data layer
    unpack LST data layer from a HDF5 container
    Args:
        hdf_file ():

    Returns:

    """
    # open the dataset
    hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
    band_ds = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly)

    # read into numpy array
    band_array = band_ds.ReadAsArray().astype(np.uint16)

    return band_array


def lst_tiff_qa(all_params):
    """

    Args:
        all_params ():

    Returns:

    """
    params, year, jd = all_params

    dir_download = params.dir_download / "modis_lst"
    os.makedirs(params.dir_interim, exist_ok=True)

    # Get the reference info (e.g., dimension, projection) for output images from the first hdf file
    hdf_files = list(dir_download.glob("*.hdf"))
    sample_file = hdf_files[0]
    sample_hdf_ds = gdal.Open(str(sample_file), gdal.GA_ReadOnly)
    sample_band_ds = gdal.Open(sample_hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly)

    XSize = sample_band_ds.RasterXSize
    YSize = sample_band_ds.RasterYSize
    GeoTransform = sample_band_ds.GetGeoTransform()
    Projection = sample_band_ds.GetProjection()

    name_file = f"MOD11C1.A{year}{str(jd).zfill(3)}_global.tif"
    path_out = params.dir_interim / name_file

    if not os.path.isfile(path_out):
        fileList = glob.glob(
            str(dir_download) + os.sep + f"MOD11C1.A{year}{str(jd).zfill(3)}*.006.*.hdf"
        )

        pbar = tqdm(fileList)
        for f in pbar:
            pbar.set_description(f)

            qa_mask = qa_extraction(f)
            outfile = hdf_subdataset_extraction(f)
            outdata = np.multiply(outfile, qa_mask)

            out_ds = gdal.GetDriverByName("GTiff").Create(
                str(path_out), XSize, YSize, 1, gdal.GDT_UInt16  # Number of bands
            )
            out_ds.SetGeoTransform(GeoTransform)
            out_ds.SetProjection(Projection)
            out_ds.GetRasterBand(1).WriteArray(outdata)
            out_ds.GetRasterBand(1).SetNoDataValue(np.NaN)
            out_ds = None


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    import itertools

    download_LST(params)

    all_params = []
    for year in range(params.start_year, params.end_year + 1):
        for jd in range(start_jd, end_jd, 1):
            all_params.extend(list(itertools.product([params], [year], [jd])))

    if False and params.parallel_process:
        with multiprocessing.Pool(
            int(multiprocessing.cpu_count() * params.fraction_cpus)
        ) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(lst_tiff_qa, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            lst_tiff_qa(val)


if __name__ == "__main__":
    pass
