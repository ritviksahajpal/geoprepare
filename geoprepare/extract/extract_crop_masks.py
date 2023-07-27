# README ##############################################################################################################
# AIM: create crop masks containing crop area percentage for each admin1 in countries of interest
# 1. Read in crop area percentage rasters in dir_per_crop_mask and crop to rasters for each admin1
#        1.a admin1 names can be found in the lookup table lup_cmask
#        1.b admin1 condaries are in ras_level1
# 2. store rasters for each admin1 in dir_crop_masks (.../GEOGLAM/Input/crop_masks)
# 3. store histograms for each admin1 in dir_crop_masks + 'hisograms' (.../GEOGLAM/Input/crop_masks/histograms)
#######################################################################################################################
import glob
import os
import pdb
import ast
from configparser import ConfigParser
from pathlib import Path

import logging
import itertools
import pandas as pd
import numpy as np
import geopandas as gp
import bottleneck as bn
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import rasterio


def get_adm_names(vals, name_val):
    """

    :param vals:
    :param name_val:
    :return:
    """
    return vals.get(name_val, default=np.nan)


def get_crop_name(long_name, use_cropland):
    """

    Args:
        long_name ():
        use_cropland ():

    Returns:

    """
    if use_cropland:
        if os.path.splitext(os.path.basename(long_name))[0] == "cropland_v9":
            return "cr"
    else:
        if os.path.splitext(os.path.basename(long_name))[0] == "Percent_Spring_Wheat":
            return "sw"
        elif os.path.splitext(os.path.basename(long_name))[0] == "Percent_Winter_Wheat":
            return "ww"
        elif os.path.splitext(os.path.basename(long_name))[0] == "Percent_Maize":
            return "mz"
        elif os.path.splitext(os.path.basename(long_name))[0] == "Percent_Soybean":
            return "sb"
        elif os.path.splitext(os.path.basename(long_name))[0] == "Percent_Rice":
            return "rc"

    return


def mask(path_raster, shape):
    import rasterio.mask

    with rasterio.open(path_raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape)

    return out_image


def create_crop_masks(params, path_crop_mask, country, scale, df_cmask):
    """

    Args:
        params ():
        path_crop_mask ():
        country ():
        df_cmask ():

    Returns:

    """
    df_cmask = df_cmask[df_cmask["lcountry"] == country]

    # Iterate though rows of dataframe, create crop masks for each region
    for row in df_cmask.iterrows():
        name_adm0 = get_adm_names(row[1], "ADM0_NAME")
        name_adm1 = get_adm_names(row[1], "ADM1_NAME")
        str_ID = get_adm_names(row[1], "str_ID")

        # Do not create masks for missing ADM1's
        if not name_adm1:
            continue

        # convert adm1 and adm0 names to standard format (lower case, no spaces and no periods
        name_adm0 = name_adm0.lower().strip().replace(" ", "_").replace(".", "")
        name_adm1 = name_adm1.lower().strip().replace(" ", "_")

        crop_name = get_crop_name(
            path_crop_mask, params.parser.getboolean(name_adm0, "use_cropland_mask")
        )
        if not crop_name:
            continue

        # Create output directory
        dir_out = params.dir_crop_masks / name_adm0 / crop_name / scale
        path_out_ras = (
            dir_out / f"{name_adm1}_{str(str_ID).zfill(9)}_{crop_name}_crop_mask.tif"
        )

        # If subset file exists, then do not recreate it
        if not os.path.isfile(path_out_ras):
            os.makedirs(dir_out, exist_ok=True)

            params.logger.info(
                f"creating crop mask in {dir_out} for {name_adm0}, {name_adm1}"
            )
            arr = mask(path_crop_mask, [row[1]["geometry"]])[0]
            arr[arr < 0] = 0.0

            # If sum of array is 0, then do not output array
            if not np.ma.sum(arr):
                continue

            with rasterio.open(path_crop_mask) as src_cmask:
                profile = src_cmask.profile
                profile.update(dtype=rasterio.int32, count=1, nodata=0, compress="lzw")

            try:
                with rasterio.open(path_out_ras, "w", **profile) as dst:
                    dst.write(arr.astype(rasterio.int32), 1)
            except:
                params.logger.error(
                    f"Cannot create crop-mask {name_adm0} {name_adm1}_{str(str_ID).zfill(9)}_{crop_name}"
                )


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    for country in params.countries:
        df_cmask = gp.GeoDataFrame.from_file(
            params.dir_regions_shp / params.parser.get(country, "shp_boundary")
        )
        df_cmask.fillna({"ADM0_NAME": "", "ADM1_NAME": ""}, inplace=True)
        df_cmask["lcountry"] = df_cmask["ADM0_NAME"].str.replace(" ", "_").str.lower()
        df_cmask = df_cmask[
            [
                "ADM1_NAME",
                "ADM0_NAME",
                "Country_ID",
                "Region_ID",
                "num_ID",
                "str_ID",
                "R_ID",
                "C_ID",
                "lcountry",
                "geometry",
            ]
        ]

        # Check if we use a cropland mask or not
        use_cropland_mask = params.parser.get(country, "use_cropland_mask")
        scales = ast.literal_eval(params.parser.get(country, "scales"))

        # Create crop masks for region
        if use_cropland_mask:
            path_mask = (
                params.dir_global_datasets
                / "masks"
                / params.parser.get(country, "mask")
            )

            for scale in scales:
                create_crop_masks(params, path_mask, country, scale, df_cmask)
        else:
            crops = ast.literal_eval(params.parser.get(country, "crops"))

            for crop in crops:
                path_mask = (
                    params.dir_global_datasets
                    / "masks"
                    / params.parser.get(crop, "mask")
                )

                for scale in scales:
                    create_crop_masks(params, path_mask, country, scale, df_cmask)


if __name__ == "__main__":
    pass
