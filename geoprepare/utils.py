import os

import datetime
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


def fill_till_last_valid(grp, fill_zero=False):
    """
    Gap fill, till current date
    If use_zero is True then fill with 0 instead of median values (default behavior)
    Args:
        grp:
        fill_zero:

    Returns:

    """
    idx = pd.Timestamp(datetime.datetime.now())

    if fill_zero:
        grp.loc[:idx] = grp.loc[:idx].astype(float).fillna(0.)
    else:
        grp.loc[:idx] = grp.loc[:idx].astype(float).fillna(grp.astype(float).median())

    return grp


def extended_dataframe(df, eo_vars):
    """

    Args:
        df ():
        eo_vars ():

    Returns:

    """
    import calendar

    select_year = 2017 if not calendar.isleap(df['year'].max() + 1) else 2018
    columns = df.columns.values.tolist()
    # Exclude eo_vars from columns
    columns = [x for x in columns if x not in eo_vars]

    # Created a new dataframe with the same columns as the original dataframe
    df_extended = df[df['year'] == select_year]

    # Fill in dataframe with NaNs for missing values
    df_extended.loc[:, df_extended.columns.difference(columns)] = np.NaN
    df_extended.loc[:, 'year'] = df['year'].max() + 1
    df_extended.index = pd.to_datetime(df_extended['year'].astype(int), format='%Y') + pd.to_timedelta(df_extended['doy'] - 1, unit='d')
    df_extended.loc[:, 'datetime'] = df_extended.index

    df.set_index(pd.DatetimeIndex(df['datetime']), inplace=True)  # revert from range to datetime index
    df = df.combine_first(df_extended)

    df.index.name = None
    df.sort_values(by=['region', 'datetime'], inplace=True)

    return df


def fill_missing_values(df_missing, eo_vars):
    """
    Fill in missing values for variable using following logic:
    1. Fill in missing values by linearly interpolating between existing values
    2. Use climatology information from the same adm1 to fill
    3. If any values are still missing, then fill using climatology (median) data from ALL adm1's
    4. Fill in any remaining missing values by forward/backward fill for 7/30 days

    Does not fill missing values for yield, crop_condition_class
    Args:
        df_missing:
        eo_vars:

    Returns:

    """
    df_missing.reset_index(inplace=True)  # change from datetime index to range to allow interpolation to happen

    # First pass at filling in missing values
    for var in eo_vars:
        # Bail if any of the following conditions are met:
        # 1. var is not in df_missing
        # 2. Column is completely empty (i.e. all years) then bail since we cannot use climatology data to fill it
        if (var not in df_missing.columns) or (df_missing[var].isnull().all()):
            continue

        # Fill in missing values by linearly interpolating between existing values for non CHIRPS datasets
        if var != 'chirps':
            df_missing.loc[:, var] = df_missing.loc[:, var].interpolate(method='linear', limit_direction='backward', limit=10).values

    # Append empty year to dataframe to account for seasons that span across years
    df_missing = extended_dataframe(df_missing, eo_vars)

    # Second pass at filling in missing values
    for var in eo_vars:
        # TODO Hack: Replace nan values with 0 for CHIRPS and CPC_PRECIP
        if var in ['chirps', 'cpc_precip']:
            continue

        # If column is completely empty (i.e. all years) then bail since we cannot use climatology data to fill it
        if df_missing[var].isnull().all():
            continue

        # Fill in any remaining missing values by forward/backward fill for 30 days
        if df_missing[var].isnull().any():
            df_missing[var].interpolate(method='linear', limit=30, inplace=True, limit_direction='backward', limit_area='inside')

    precip_var = 'chirps' if 'chirps' in eo_vars else 'cpc_precip'
    df_missing[precip_var] = df_missing.groupby(df_missing['region'])[precip_var].transform(fill_till_last_valid, fill_zero=True)

    if not isinstance(df_missing.index, pd.DatetimeIndex):
        raise ValueError('dataframe index is incorrect')

    return df_missing


def remove_leap_doy(df):
    """
    Remove day corresponding to Feb 29th from dataframe.
    Dataframe can span across multiple years and may or may not include leap year
    For Julian day column, reduce julian day value by 1 for each day following Feb 29th
    Args:
        df:

    Returns:

    """
    # Remove row for Feb 29th
    df = df[~is_leap(df)]

    # Recompute JD since Feb 29th has been removed now
    frames = []
    groups = df.groupby(['region', 'year'])
    for key, vals in groups:
        vals['doy'] = range(1, len(vals) + 1)
        frames.append(vals)

    df = pd.concat(frames)

    return df

def is_leap(s):
    """
    Assume that index is a datetime object
    Args:
        s: dataframe with datetime index

    Returns: If year is leap or not (True/False)

    """
    if isinstance(s.index, pd.DatetimeIndex):
        return (s.index.year % 4 == 0) & ((s.index.year % 100 != 0) | (s.index.year % 400 == 0)) & \
               (s.index.month == 2) & (s.index.day == 29)
    else:
        raise ValueError('Index should be a datetime object')
