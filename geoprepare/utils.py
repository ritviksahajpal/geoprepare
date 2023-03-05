import os

import rasterio
import xarray as xr
import numpy as np
import pandas as pd


def read_config(path_config_file):
    """

    Args:
        path_config_file ():

    Returns:

    """
    from configparser import ConfigParser, ExtendedInterpolation
    parser = ConfigParser(inline_comment_prefixes=(';',), interpolation=ExtendedInterpolation())

    try:
        parser.read(path_config_file)
    except Exception as e:
        raise IOError(f'Cannot read {path_config_file}: {e}')

    return parser


def unzip_file(path_file):
    """
    Unzips a file
    """
    if path_file.endswith('.gz'):
        os.system(f"gunzip {path_file}")


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


def arr_to_tif(arr, path_tif, profile):
    """

    :param arr:
    :param path_tif:
    :param profile:
    :return:
    """
    with rasterio.open(path_tif, 'w', **profile) as dst:
        dst.write(arr, 1)


def crop_mask_limit(params, country, threshold):
    """

    Args:
        params ():
        country ():
        threshold ():

    Returns:

    """
    limit_type = 'floor' if threshold else 'ceil'

    limit = params.parser.getint(country, limit_type)

    return limit


def harmonize_df(df, columns=None):
    """

    Args:
        df ():
        columns ():

    Returns:

    """
    if columns is None:
        columns = df.columns

    for col in columns:
        # Check if column contains string values, if yes then convert to lower case and replace spaces with underscores
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(' ', '_').str.lower()

    return df


def get_cal_list(df_cal, yr_cal):
    """
    Convert from GEOGLAM calendar to a pandas daily time-series
    Example input:
       jan_1  jan_15  feb_1  feb_15  mar_1  mar_15  ...
    5     -1      -1     -1      -1     -1      -1

    Example output:
                 col
    2000-01-01   0
    2000-01-02   0
    2000-01-03   1
    2000-01-04   1
    Args:
        df_cal:
        yr_cal: e.g. 2000

    Returns:

    """
    idf_cal = df_cal.copy()

    # Name the last column as dec_31
    idf_cal['dec_31'] = idf_cal.iloc[:, -1]

    # converting from jan_1 to 2000-01-01
    # Rename all columns
    idf_cal.rename(columns=lambda x: str(yr_cal) + '_' + x, inplace=True)

    # convert to datetime - see http://strftime.org/
    idf_cal.columns = pd.to_datetime(idf_cal.columns, format='%Y_%b_%d')

    # transpose (convert from horizontal to vertical) and resample from biweekly to daily
    idf_cal = (idf_cal
               .T
               .resample('d').ffill()
               .rename(columns=lambda x: 'col'))

    # Drop if the column is all -1
    idf_cal = idf_cal.loc[:, (idf_cal != -1).any(axis=0)]

    return idf_cal
