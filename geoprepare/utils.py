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

    parser = ConfigParser(
        inline_comment_prefixes=(";",), interpolation=ExtendedInterpolation()
    )

    try:
        parser.read(path_config_file)
    except Exception as e:
        raise IOError(f"Cannot read {path_config_file}: {e}")

    return parser


def unzip_file(path_file):
    """
    Unzips a file
    """
    if path_file.endswith(".gz"):
        os.system(f"gunzip {path_file}")


def convert_to_nc_hndl(path_nc):
    """

    :param path_nc:
    :return:
    """
    hndl_nc = path_nc
    if not isinstance(path_nc, np.ma.MaskedArray):
        _, ext = os.path.splitext(path_nc)

    if ext in [".nc", ".nc4"]:
        hndl_nc = xr.open_dataset(path_nc)

    return hndl_nc


def arr_to_tif(arr, path_tif, profile):
    """

    :param arr:
    :param path_tif:
    :param profile:
    :return:
    """
    with rasterio.open(path_tif, "w", **profile) as dst:
        dst.write(arr, 1)


def crop_mask_limit(params, country, threshold):
    """

    Args:
        params ():
        country ():
        threshold ():

    Returns:

    """
    limit_type = "floor" if threshold else "ceil"

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
        if df[col].dtype == "object":
            df[col] = df[col].str.replace(" ", "_").str.lower()

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
    idf_cal["dec_31"] = idf_cal.iloc[:, -1]

    # converting from jan_1 to 2000-01-01
    # Rename all columns
    idf_cal.rename(columns=lambda x: str(yr_cal) + "_" + x, inplace=True)

    # convert to datetime - see http://strftime.org/
    idf_cal.columns = pd.to_datetime(idf_cal.columns, format="%Y_%b_%d")

    # transpose (convert from horizontal to vertical) and resample from biweekly to daily
    idf_cal = idf_cal.T.resample("d").ffill().rename(columns=lambda x: "col")

    # Drop if the column is all -1
    idf_cal = idf_cal.loc[:, (idf_cal != -1).any(axis=0)]

    return idf_cal


def fill_till_valid(group, fill_zero=False, interpolate=False):
    """
    Gap fill, till current date
    If use_zero is True then fill with 0 instead of median values (default behavior)
    Args:
        group:
        fill_zero:
        interpolate:

    Returns:

    """
    # Assert that both fill_zero and interpolate are not True at the same time
    assert not (
        fill_zero and interpolate
    ), "fill_zero and interpolate cannot be True at the same time"
    # Assert that both fill_zero and interpolate are not False at the same time
    assert not (
        not fill_zero and not interpolate
    ), "fill_zero and interpolate cannot be False at the same time"

    # The last date for which we can have valid data is today's date
    idx = pd.Timestamp(datetime.datetime.now())

    if fill_zero:
        group.loc[:idx] = group.loc[:idx].astype(float).fillna(0.0)
    elif not fill_zero and not interpolate:
        group.loc[:idx] = (
            group.loc[:idx].astype(float).fillna(group.astype(float).median())
        )
    elif not fill_zero and interpolate:
        group.loc[:idx] = (
            group.loc[:idx]
            .astype(float)
            .interpolate(
                method="linear",
                limit_direction="backward",
                limit_area="inside",
                limit=30,
            )
        )
    else:
        raise NotImplementedError(
            f"fill_zero: {fill_zero} and interpolate: {interpolate} not implemented"
        )

    return group


def extended_dataframe(df, eo_vars):
    """

    Args:
        df ():
        eo_vars ():

    Returns:

    """
    import calendar

    select_year = 2017 if not calendar.isleap(df["year"].max() + 1) else 2016

    columns = df.columns.values.tolist()
    # Exclude eo_vars from columns
    columns = [x for x in columns if x not in eo_vars]

    # Created a new dataframe with the same columns as the original dataframe
    df_extended = df[df["year"] == select_year]

    # Fill in dataframe with NaNs for missing values
    df_extended.loc[:, df_extended.columns.difference(columns)] = np.NaN
    df_extended.loc[:, "year"] = df["year"].max() + 1
    df_extended.index = pd.to_datetime(
        df_extended["year"].astype(int), format="%Y"
    ) + pd.to_timedelta(df_extended["doy"] - 1, unit="d")
    df_extended.loc[:, "datetime"] = df_extended.index

    df.set_index(
        pd.DatetimeIndex(df["datetime"]), inplace=True
    )  # revert from range to datetime index
    df = df.combine_first(df_extended)

    df.index.name = None
    df.sort_values(by=["region", "datetime"], inplace=True)

    return df


def fill_missing_values(df, eo_vars):
    """
    Fill in missing values for variable using following logic:
    1. Fill in missing values by linearly interpolating between existing values
    2. Use climatology information from the same adm1 to fill
    3. If any values are still missing, then fill using climatology (median) data from ALL adm1's
    4. Fill in any remaining missing values by forward/backward fill for 7/30 days

    Does not fill missing values for yield, crop_condition_class
    Args:
        df:
        eo_vars:

    Returns:

    """
    precip_var = "chirps" if "chirps" in eo_vars else "cpc_precip"

    # Append empty year to dataframe to account for seasons that span across years
    df = extended_dataframe(df, eo_vars)

    # First pass at filling in missing values
    for var in eo_vars:
        # Bail if any of the following conditions are met:
        # 1. var is not in df
        # 2. Column is completely empty (i.e. all years) then bail since we cannot use climatology data to fill it
        if var not in df.columns:
            continue

        if var == precip_var:
            df[var] = df.groupby(df["region"])[var].transform(
                fill_till_valid, fill_zero=True
            )
        else:
            df[var] = df.groupby(df["region"])[var].transform(
                fill_till_valid, fill_zero=False, interpolate=True
            )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("dataframe index is incorrect")

    return df


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
    groups = df.groupby(["region", "year"])
    for key, vals in groups:
        vals["doy"] = range(1, len(vals) + 1)
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
        return (
            (s.index.year % 4 == 0)
            & ((s.index.year % 100 != 0) | (s.index.year % 400 == 0))
            & (s.index.month == 2)
            & (s.index.day == 29)
        )
    else:
        raise ValueError("Index should be a datetime object")


def mosaic(tif_files, output_file):
    """
    Merges multiple TIFF files into a single mosaic TIFF file.

    Args:
        tif_files (list): List of file paths to the input TIFF files.
        output_file (str): File path to the output mosaic TIFF file.

    Returns:
        None
    """
    import rasterio
    from rasterio.merge import merge
    from concurrent.futures import ThreadPoolExecutor

    def open_file(fp):
        """Opens a raster file."""
        return rasterio.open(fp)

    # Read the tif files in parallel
    with ThreadPoolExecutor() as executor:
        src_files_to_mosaic = list(executor.map(open_file, tif_files))

    # Mosaic the files
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Copy the metadata
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update the metadata for mosaic
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],  # Height of the mosaic
            "width": mosaic.shape[2],  # Width of the mosaic
            "transform": out_trans,
            "count": mosaic.shape[0],  # Number of bands
        }
    )

    # Write the mosaic raster to disk
    with rasterio.open(output_file, "w", **out_meta) as dest:
        for i in range(1, mosaic.shape[0] + 1):  # Write each band
            dest.write(mosaic[i - 1], i)

    # Close the files
    for src in src_files_to_mosaic:
        src.close()
