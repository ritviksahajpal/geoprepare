import os

import rasterio
import xarray as xr
import numpy as np


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
