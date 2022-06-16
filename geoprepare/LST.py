# README ##############################################################################################################
# Data Preprocessing
# Global Daily MODIS Land Surface Temperature (LST) data at 0.05Deg (MOD11C1.006)
# The data is already provided at a 0.05 degree resolution, global extent to match the crop masks.
# For each day, pixels with poor LST quality were masked out based on the quality assurance (QA) layer (Note: The two least significant bits "00" indicate LST produced with good quality).
# Daily LST data with good quality were output into daily tiff file.
# Values range between 7500-65535. Fill values are 0. Fill values are masked out in the Weighted Average Extraction code.
# The value is scaled Kelvin temperature and can be converted to Celsius temperature by LST=DN*0.02-273.15(°C).
#######################################################################################################################
import os
import always
from osgeo import gdal
import numpy as np
import glob
import pymodis
import multiprocessing
from tqdm import tqdm
import pygeoutil.util as util

import Code.preprocess.constants_preprocess as constants
import Code.base.log as log
import Code.base.constants as cc

logger = log.Logger(dir_log=cc.dir_tmp, name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0])

syr = 2000
eyr = constants.END_YEAR
sjd = constants.START_JD
ejd = constants.END_JD


def download_LST():
    # destination folder
    dir_download = constants.dir_download / 'modis_lst'
    util.make_dir_if_missing(dir_download)

    # number of days to check for update, count backward from the most recent data
    count = 15

    modisDown = pymodis.downmodis.downModis(destinationFolder=dir_download,
                                            url="https://e4ftl01.cr.usgs.gov", tiles=None, path="MOLT",
                                            product="MOD11C1.006", delta=count)
    modisDown.connect()
    count = len(modisDown.getListDays())

    pbar = tqdm(range(0, count, 1))
    for n in pbar:
        day = modisDown.getListDays()[n]
        pbar.set_description(f'Download MODIS LST for {day}')

        # get the list of data files for specific day
        list_new_file = modisDown.getFilesList(day)

        # check whether the data exists and return only the list of previously non-downloaded data files
        list_files_down = modisDown.checkDataExist(list_new_file, move=False)

        # download the previously non-downloaded data file
        if modisDown.nconnection <= 20:
            # maximum number of attempts to connect to the HTTP server before failing
            modisDown.dayDownload(day, list_files_down)
        else:
            logger.error("A problem with the connection occured")


def qa_extraction(qa_infile):
    """unpack qa data layer from a HDF5 container """

    # open the dataset
    qa_hdf_ds = gdal.Open(qa_infile, gdal.GA_ReadOnly)
    qa_ds = gdal.Open(qa_hdf_ds.GetSubDatasets()[1][0], gdal.GA_ReadOnly)

    # read into numpy array
    qa_array = qa_ds.ReadAsArray().astype(np.uint8)

    # read qa bit info, get the 2 least significant bits for quality check ('00' means good LST quality)
    mask = ((qa_array & 3) == 0)

    # mask pixels without good LST quality as NaN
    mask[mask == 0.0] = np.NaN

    return mask


def hdf_subdataset_extraction(hdf_file):
    # Retrieve the LST data layer
    """unpack LST data layer from a HDF5 container """

    # open the dataset
    hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
    band_ds = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly)

    # read into numpy array
    band_array = band_ds.ReadAsArray().astype(np.uint16)

    return band_array


def lst_tiff_qa(all_params):

    year, jd = all_params[0], all_params[1]

    dir_download = constants.dir_download / 'modis_lst'
    dir_out = constants.dir_intermed / 'lst'
    util.make_dir_if_missing(dir_out)

    # Get the reference info (e.g., dimension, projection etc) for output images from a sample file
    sample_file = dir_download / 'MOD11C1.A2018226.006.2018227181437.hdf'
    sample_hdf_ds = gdal.Open(str(sample_file), gdal.GA_ReadOnly)
    sample_band_ds = gdal.Open(sample_hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly)

    XSize = sample_band_ds.RasterXSize
    YSize = sample_band_ds.RasterYSize
    GeoTransform = sample_band_ds.GetGeoTransform()
    Projection = sample_band_ds.GetProjection()

    name_file= 'MOD11C1.A' + str(year) + str(jd).zfill(3) + '_global.tif'
    path_out = dir_out / name_file

    if not os.path.isfile(path_out):

        fileList = glob.glob(str(dir_download) + os.sep + 'MOD11C1.A' + str(year) + str(jd).zfill(3) + '*.006.*.hdf')

        pbar = tqdm(fileList)
        for f in pbar:
            pbar.set_description(f)

            qa_mask = qa_extraction(f)
            outfile = hdf_subdataset_extraction(f)
            outdata = np.multiply(outfile, qa_mask)

            out_ds = gdal.GetDriverByName('GTiff').Create(str(path_out),
                                                          XSize,
                                                          YSize,
                                                          1,  # Number of bands
                                                          gdal.GDT_UInt16)
            out_ds.SetGeoTransform(GeoTransform)
            out_ds.SetProjection(Projection)
            out_ds.GetRasterBand(1).WriteArray(outdata)
            out_ds.GetRasterBand(1).SetNoDataValue(np.NaN)
            out_ds = None

    else:
        pass
        # logger.info(f'File exists: {dir_out} {name_file}')


if __name__ == '__main__':
    import itertools

    # Store git hash of current code
    logger.info('################ GIT HASH ################')
    logger.info(util.get_git_revision_hash())
    logger.info('################ GIT HASH ################')

    download_LST()

    all_params = []
    for year in range(syr, eyr):
        for jd in range(sjd, ejd, 1):
            all_params.extend(list(itertools.product([year], [jd])))

    if constants.do_parallel_processing:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(lst_tiff_qa, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            lst_tiff_qa(val)