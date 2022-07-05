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
    :param long_name:
    :return:
    """
    if use_cropland:
        if os.path.splitext(os.path.basename(long_name))[0] == 'cropland_v9':
            return 'cr'
    else:
        if os.path.splitext(os.path.basename(long_name))[0] == 'Percent_Spring_Wheat':
            return 'sw'
        elif os.path.splitext(os.path.basename(long_name))[0] == 'Percent_Winter_Wheat':
            return 'ww'
        elif os.path.splitext(os.path.basename(long_name))[0] == 'Percent_Maize':
            return 'mz'
        elif os.path.splitext(os.path.basename(long_name))[0] == 'Percent_Soybean':
            return 'sb'
        elif os.path.splitext(os.path.basename(long_name))[0] == 'Percent_Rice':
            return 'rc'

    return


def mask(path_raster, shape):
    import fiona
    import rasterio.mask

    with rasterio.open(path_raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)

    return out_image


def create_crop_masks(params, path_crop, country, df_cmask):
    """

    Args:
        params ():
        path_crop ():
        country ():
        df_cmask ():

    Returns:

    """
    df = pd.DataFrame(columns=['adm0', 'adm1', 'crop', f'p{params.upper_percentile}'])
    df_cmask = df_cmask[df_cmask['lcountry'] == country]

    # Iterate though rows of dataframe, create crop masks for each ADM1 region inside of a folder named after ADM0
    # Read lookup table that links admin1's to the admin0 or country that they are in
    for row in df_cmask.iterrows():
        name_adm0 = get_adm_names(row[1], 'ADM0_NAME')
        name_adm1 = get_adm_names(row[1], 'ADM1_NAME')

        str_ID = get_adm_names(row[1], 'str_ID')
        num_ID = str(get_adm_names(row[1], 'num_ID'))

        name_adm0 = name_adm0.lower().strip().replace(' ', '_').replace('.', '')
        name_adm1 = name_adm1.lower().strip().replace(' ', '_')

        # Do not create masks for missing ADM1's
        if not name_adm1:
            continue

        # Create output directory
        dir_crop = get_crop_name(path_crop, params.parser.getboolean(name_adm0, 'use_cropland_mask'))
        if not dir_crop:
            continue

        dir_out = params.dir_crop_masks / name_adm0 / dir_crop
        path_out_ras = dir_out / f'{name_adm1}_{str(str_ID).zfill(9)}_{dir_crop}_crop_mask.tif'

        # If subset file exists, then do not recreate it
        if not os.path.isfile(path_out_ras):
            os.makedirs(dir_out, exist_ok=True)

            # params.logger.info(f'{dir_out} {name_adm0} {name_adm1}')

            # Open the global .tif file and split into ADM1s
            # with rasterio.open(dat_level1) as src:
            #     b1 = src.read(1)
            #
            pdb.set_trace()
            b1 = mask(path_crop, [row['geometry']])

            # Copy data to new array and replace all values except those corresponding to ADM1_NAME
            arr = b1.copy()
            arr[arr != int(num_ID)] = 0
            arr[arr != 0] = 1

            # Mask by crop percentage mask
            # https://github.com/ozak/georasters
            with rasterio.open(path_crop) as src_cmask:
                b2 = src_cmask.read(1)
                arr = arr * b2  # Multiply masked region with crop mask containing percentage of crop

                profile = src_cmask.profile
                profile.update(
                    dtype=rasterio.int32,
                    count=1,
                    nodata=0,
                    compress='lzw')

                # If sum of array is 0, then do not output array
                if not np.ma.sum(arr):
                    continue

            try:
                with rasterio.open(path_out_ras, 'w', **profile) as dst:
                    dst.write(arr.astype(rasterio.int32), 1)
            except:
                params.logger.error(f'Cannot create crop-mask {name_adm0} {name_adm1}_{str(str_ID).zfill(9)}_{dir_crop}')

    return df


def run(params):
    crops = params.dir_masks.glob('*.tif')

    for country in params.countries:
        category = params.parser.get(country, 'category')

        df_cmask = gp.GeoDataFrame.from_file(params.dir_regions_shp / params.parser.get(country, 'shp_boundary'))
        df_cmask.fillna({'ADM0_NAME': '', 'ADM1_NAME': ''}, inplace=True)
        df_cmask['lcountry'] = df_cmask['ADM0_NAME'].str.replace(' ', '_').str.lower()
        df_cmask = df_cmask[['ADM1_NAME', 'ADM0_NAME', 'Country_ID', 'Region_ID', 'num_ID', 'str_ID', 'R_ID', 'C_ID', 'lcountry', 'geometry']]

        frames = []
        for crop in tqdm(crops):
            df = create_crop_masks(params, crop, country, df_cmask)
            df.loc[:, 'category'] = category

            frames.append(df)

    if len(frames):
        df_full = pd.concat(frames)
        df_full.to_csv(params.dir_crop_masks / 'stats_crop_masks.csv')


if __name__ == '__main__':
    # loop_get_crop_area()
    run()
