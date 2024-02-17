####################################
# Author: F. Dan O'Neill
# Date: 1/23/2020
####################################
"""This script takes a product name and a directory, and
checks for newly available imagery or gaps in the archive.
Imagery is then downloaded and written to disk to complete
the archive.

Call is:

python update_cmg_archive.py {PRODUCT} {DIRECTORY} {scale_flag}

e.g.

python update_cmg_archive.py "MOD09CMG" "C:/terra_archive/" -scale_mark

***

Parameters
----------
product:str
	Name of desired imagery product
directory:str
	Path to directory where imagery
	is currently stored

Flags
-----
-scale_glam
	Use GLAM system NDVI scaling
	(NDVI * 10000)
-scale_mark
	Use Mark's NDVI scaling
	((NDVI * 200) + 50)
-start_year {%Y}
	If this flag is not set, the script
	sets the start year to that of the
	earliest existing file in the directory.
	This flag overrides that method, which is
	necessary when first filling a directory,
	for example.

"""

## set up logging
import logging, os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

## import modules
import argparse, glob, octvi, shutil, sys
from osgeo import gdal
from osgeo.gdal_array import *
import numpy as np
from datetime import datetime, timedelta


def generateFileName(
    product: str, vi: str, year: int = None, doy: int = None, date: str = None
) -> str:
    """This function converts a date into a valid file name

    Specify EITHER 'year' and 'doy' OR 'date'

    ***

    Parameters
    ----------
    product:str
        Name of imagery product (e.g. "MOD09CMG")
    year:int
        Year of image
    doy:int
        Day of year (doy) of image
    date:str
        %Y-%m-%d
    """
    if year is None or doy is None:
        try:
            year, doy = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%j").split("-")
        except:
            log.error("Specify either year and doy OR date!")

    prefix = product[:5].lower()
    if vi == "ndvi":
        return f"{prefix}.ndvi.global_0.05_degree.{year}.{str(doy).zfill(3)}.c6.v1.tif"
    elif vi == "gcvi":
        return f"{prefix}.gcvi.global_0.05_degree.{year}.{str(doy).zfill(3)}.c6.v1.tif"


def check8DayValidity(product: str, date: str) -> bool:
    """This function returns whether all files in an 8-day range from
    the provided date are available.

    ***

    Parameters
    ---------
    product:str
        e.g. "MOD09CMG"
    date:str
        %Y-%m-%d
    """
    for i in range(0, 9):
        compDate = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=i)).strftime(
            "%Y-%m-%d"
        )
        if len(octvi.url.getDates(product, compDate)) == 0:
            return False
    return True


def scaleConversion_glamToMark(in_file: str) -> None:
    """This function changes the scaling on an input file

    The values are converted from GLAM scaling (NDVI * 10000) to
    Mark's scaling ((NDVI * 200) + 50)

    ***

    Parameters
    ----------
    in_file:str
        Path to the image file to be converted

    """
    ## define intermediate filename
    extension = os.path.splitext(in_file)[1]
    intermediate_file = in_file.replace(extension, f".TEMP{extension}")
    ## copy to intermediate file
    with open(in_file, "rb") as rf:
        with open(intermediate_file, "wb") as wf:
            shutil.copyfileobj(rf, wf)
    ## open file with gdal
    ds = gdal.Open(in_file, 0)
    # extract projection and geotransform
    srs = ds.GetProjection()
    gt = ds.GetGeoTransform()
    # convert to array
    ds_band = ds.GetRasterBand(1)
    arr = BandReadAsArray(ds_band)
    rasterYSize, rasterXSize = arr.shape
    # close file and band
    ds = ds_band = None

    ## convert array values
    arr = ((arr * .02) + 50)
    arr[arr == -10] = 255
    arr = np.uint8(arr)

    ## write to file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(in_file, rasterXSize, rasterYSize, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])
    dataset.GetRasterBand(1).WriteArray(arr)
    dataset.GetRasterBand(1).SetNoDataValue(255)
    dataset.SetGeoTransform(gt)
    dataset.SetProjection(srs)
    dataset.FlushCache()  # Write to disk
    del dataset

    ## remove intermediate file
    os.remove(intermediate_file)


def run(params):
    from tqdm import tqdm

    dir_interim = params.dir_interim / "ndvi"
    os.makedirs(dir_interim, exist_ok=True)

    ## validate arguments
    if params.scale_glam:
        assert not params.scale_mark, "Set exactly one of -scale_glam or -scale_mark!"
    else:
        assert params.scale_mark, "Set exactly one of -scale_glam or -scale_mark!"
        assert params.vi == "ndvi", "MARK scaling should only be used with NDVI"

    ## clean and collect existing files
    # remove running composites
    runningComposites = glob.glob(os.path.join(dir_interim, "*.running_composite.tif"))
    for c in runningComposites:
        os.remove(c)
    # get list of existing full files
    extantFiles = glob.glob(os.path.join(dir_interim, f"*.tif"))

    ## get missing dates
    # first and last year
    if params.start_year:
        firstYear = int(params.start_year)
    else:
        firstYear = int(datetime.now().strftime("%Y"))
        for f in extantFiles:
            try:
                fileYear = int(os.path.basename(f).split(".")[4])
                if fileYear < firstYear:
                    firstYear = fileYear
            except ValueError:
                continue
    # range of years and doys
    years = [y for y in range(firstYear, int(datetime.now().strftime("%Y")) + 1)]
    doys = [d for d in range(1, 365, 8)]

    # filter missing dates
    dates = {}
    earliestDataDict = {"MOD09CMG": "2000.055", "MYD09CMG": "2002.185"}

    for y in tqdm(years, desc="year"):
        for doy in tqdm(doys, desc="doy", leave=False):
            if not os.path.exists(
                os.path.join(
                    dir_interim, generateFileName(params.product, params.vi, y, doy)
                )
            ):
                formattedDate = datetime.strptime(f"{y}.{doy}", "%Y.%j").strftime(
                    "%Y-%m-%d"
                )
                # is the image in the future?
                if datetime.strptime(
                    f"{y}.{doy}", "%Y.%j"
                ) >= datetime.now() or datetime.strptime(
                    f"{y}.{doy}", "%Y.%j"
                ) < datetime.strptime(
                    earliestDataDict.get(params.product, "2000.049"), "%Y.%j"
                ):
                    continue
                # does the image exist at all?
                if (
                    len(octvi.url.getDates(params.product, formattedDate)) == 0
                ):  # if there is valid imagery for the date
                    dates[formattedDate] = "No imagery"
                # are there missing files in the compositing period?
                elif not check8DayValidity(params.product, formattedDate):
                    dates[formattedDate] = "Incomplete compositing period"
                # if none of the above, it's available.
                else:
                    dates[formattedDate] = "Available"
    # special behavior if "-print_missing" is set
    if params.print_missing:
        for k in dates.keys():
            outString = f"{k} : {dates[k]}"
            print(outString)
        sys.exit()

    if params.vi == "ndvi":
        availableFiles = 0
        completedFiles = 0
        for d in dates.keys():
            if dates[d] == "Available":
                availableFiles += 1
                params.logger.info(f"Creating Composite for {d}")
                outPath = os.path.join(
                    dir_interim, generateFileName(params.product, params.vi, date=d)
                )
                octvi.globalVi(params.product, d, outPath)
                if params.scale_mark:
                    scaleConversion_glamToMark(outPath)

                ## reproject to WGS 1984
                srs = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
                ds = gdal.Open(outPath, 1)
                if ds:
                    res = ds.SetProjection(srs)
                    if res != 0:
                        params.logger.error("--projection failed: {}".format(str(res)))
                        ds = None
                        continue
                    ds = None
                    completedFiles += 1
                else:
                    params.logger.error("--could not open with GDAL")
    elif params.vi == "gcvi":
        availableFiles = 0
        completedFiles = 0
        for d in dates.keys():
            if dates[d] == "Available":
                availableFiles += 1
                params.logger.info(f"Creating Composite for {d}")
                outPath = os.path.join(
                    dir_interim, generateFileName(params.product, params.vi, date=d)
                )
                octvi.globalVi(params.product, d, outPath, vi="GCVI")

                ## reproject to WGS 1984
                srs = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
                ds = gdal.Open(outPath, 1)
                if ds:
                    res = ds.SetProjection(srs)
                    if res != 0:
                        params.logger.error("--projection failed: {}".format(str(res)))
                        ds = None
                        continue
                    ds = None
                    completedFiles += 1
                else:
                    params.logger.error("--could not open with GDAL")

    params.logger.info(
        f"Done. {availableFiles} composites available; {completedFiles} composites created. Use -print_missing flag to see details of missing composites."
    )


if __name__ == "__main__":
    pass
