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

import always
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

import Code.base.log as log
import Code.base.constants as constants
import pygeoutil.util as utils
import pygeoutil.rgeo as rgeo

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', nargs='?', default='config_GEOGLAM.txt')
parser.add_argument('--season', nargs='?', type=int, default=1)
parser.add_argument('--ignore_fall', action='store_false')  # store_false will default to True when the command-line argument is not present
parser.add_argument('--output_dir', nargs='?', default='processed')
args = parser.parse_args()

parser = ConfigParser(inline_comment_prefixes=(';',))
parser.read(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'config' + os.sep + args.config)

# Logging
logger = log.Logger(dir_log=constants.dir_tmp, name_fl=os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0], level=logging.ERROR)

dir_crop_masks = constants.dir_all_inputs / 'crop_masks'

def get_cropped_area(path_crop):
    """
    Get cropped area for specific crop in an ADM1
    :param path_crop: path to crop mask
    :return: pandas dataframe containing adm0, adm1, area
    """
    df_area = pd.DataFrame(columns=['Adm0', 'Adm1', 'Adm2', 'Area'])

    # Read lookup table and create dataframe
    # shp_boundaries
    df_cmask = gp.GeoDataFrame.from_file(constants.shp_boundaries)
    df_cmask.fillna('', inplace=True)

    # Iterate though rows of dataframe, create crop masks for each ADM1 region inside of a folder named after ADM0
    # Mask by crop percentage mask
    with rasterio.open(path_crop) as src_cmask:
        per_cmask = src_cmask.read(1)
        per_cmask[per_cmask <= 500] = 0.0  # ToDo HACK Alert! 500 or 5% has been hard coded, need better way
        arr_area = rgeo.get_grid_cell_area(per_cmask.shape[0], per_cmask.shape[1])
        per_cmask = per_cmask * arr_area * constants.PERC_TO_FRAC

        profile = src_cmask.profile
        tmp_dir = os.path.dirname(path_crop) + os.sep + 'threshold'
        utils.make_dir_if_missing(tmp_dir)
        with rasterio.open(tmp_dir + os.sep + 'threshold_' + os.path.basename(path_crop), 'w', **profile) as dst:
            dst.write(per_cmask.astype(rasterio.uint16), 1)

        return
        # Open the global .tif file and split into ADM1s
        with rasterio.open(constants.ras_level1) as src:
            ras_id_reg = src.read(1)

        _area = rgeo.get_grid_cell_area(ras_id_reg.shape[0], ras_id_reg.shape[1])

        for row in df_cmask.iterrows():
            name_adm0 = get_adm_names(row[1], 'ADM0_NAME')
            name_adm1 = get_adm_names(row[1], 'ADM1_NAME')
            name_adm2 = get_adm_names(row[1], 'ADM2_NAME')

            # if admin2 does not exist, then set it to be the same as admin1
            if not len(name_adm2):
                name_adm2 = name_adm1

            id_reg = str(get_adm_names(row[1], 'num_ID'))

            # Do not create masks or compute area for missing ADM1's, but ADM2 should be blank
            if not name_adm1:
                continue
            # Remove '.' from name and replace ' ' by '_'
            # name_adm0 = name_adm0.lower().strip().replace(' ', '_').replace('.', '')
            # name_adm1 = name_adm1.lower().strip().replace(' ', '_')

            # Copy data to new array and replace all values except those corresponding to ADM1_NAME
            arr = ras_id_reg.copy()
            arr[arr != int(id_reg)] = 0
            arr[arr != 0] = 1

            arr = arr * per_cmask * _area # Multiply masked region with crop mask (% of crop) and grid cell area

            # If sum of array is 0, then do not output array
            arr_sum = bn.nansum(arr)
            if not arr_sum:
                continue

            df_area.loc[len(df_area)] = [name_adm0, name_adm1, name_adm2, arr_sum * constants.PERC_TO_FRAC * 1e-2]
            logger.info(name_adm1, name_adm0, arr_sum * constants.PERC_TO_FRAC * 1e-2)

    return df_area


def loop_get_crop_area():
    crops = glob.glob(constants.dir_per_crop_mask + os.sep + '*.tif')
    frames = pd.DataFrame()

    if not constants.do_parallel:
        for cr in crops:
            df = get_cropped_area(cr)
            frames = pd.concat([df, frames])
    else:
        pool = multiprocessing.Pool(constants.ncpu)
        df = pool.map(get_cropped_area, [crops[0]])
        pool.close()
        pool.join()

        df.to_csv(constants.dir_cmask_lup + os.sep + 'crop_area.csv')


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


def plot_hist(arr, out_dir, name_adm1, name_crop, perc=90):
    """

    Args:
        arr:
        out_dir:
        name_adm1:
        name_crop:
        perc:

    Returns:

    """
    out_fname = name_adm1 + '.tif'
    utils.make_dir_if_missing(out_dir)

    # Compute percentile
    val_perc = np.percentile(arr[arr > 0.], perc)
    val_median = np.median(arr[arr > 0.0])

    # Plot histogram of arr
    hist, bins = np.histogram(arr[arr > 0.], bins=100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots()
    ax.bar(center, hist, align='center', width=width)

    # plot vertical lines at different percentiles and values
    plt.axvline(1000, color='brown', linestyle='dashed', linewidth=1, label='Percentile: 10')
    plt.axvline(2000, color='cyan', linestyle='dashed', linewidth=1, label='Percentile: 20')
    plt.axvline(val_perc, color='g', linestyle='dashed', linewidth=1, label='Percentile: ' + str(perc))
    if val_perc != val_median:
        plt.axvline(val_median, color='red', linestyle='dashed', linewidth=1, label='Median')

    ax.grid(which='major', alpha=0.5, linestyle='--')
    plt.title(f'{name_adm1.title()} {name_crop}')
    plt.xlabel('Percentage of crop in grid-cell (x 100)')
    plt.ylabel('Frequency')

    leg = plt.legend(fancybox=None, prop={'size': 'x-small'})
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.5)

    plt.tight_layout()
    plt.savefig(out_dir / out_fname, dpi=250)
    plt.close()

    return val_perc


def get_country_name(country):
    out_name = None

    if country == 'vietnam':
        out_name = 'viet_nam'

def create_crop_masks(path_crop, country, df_cmask, dat_level1, use_adm2=False):
    """
    :param path_crop:
    :return:
    """
    df = pd.DataFrame(columns=['adm0', 'adm1', 'crop', 'p' + str(constants.upper_percentile)])
    if country == 'vietnam':
        df_cmask = df_cmask[df_cmask['lower_ADM0_NAME'] == 'viet_nam']
    else:
        df_cmask = df_cmask[df_cmask['lower_ADM0_NAME'] == country]

    # Iterate though rows of dataframe, create crop masks for each ADM1 region inside of a folder named after ADM0
    # Read lookup table that links admin1's to the admin0 or country that they are in
    for row in df_cmask.iterrows():
        name_adm0 = get_adm_names(row[1], 'ADM0_NAME')
        name_adm1 = get_adm_names(row[1], 'ADM1_NAME')
        name_adm0 = name_adm0.replace('ô', 'o')

        if name_adm0 == 'Viet Nam':
            name_adm0 = 'vietnam'

        # if admin2 does not exist, then set it to be the same as admin1
        if use_adm2:
            name_adm2 = get_adm_names(row[1], 'ADM2_NAME')
            if not len(name_adm2):
                name_adm2 = name_adm1

        # Create crop mask only if adm0 is in country list
        if name_adm0.lower() not in [x.replace('_', ' ').lower() for x in constants.COUNTRIES]:
            logger.info('Ignoring ' + name_adm0)
            continue

        str_ID = get_adm_names(row[1], 'str_ID')
        num_ID = str(get_adm_names(row[1], 'num_ID'))

        # if name_adm0 != 'United States of America' and name_adm0 != 'Russian Federation':
        #    continue

        name_adm0 = name_adm0.lower().strip().replace(' ', '_')
        name_adm1 = name_adm1.lower().strip().replace(' ', '_')

        # Do not create masks for missing ADM1's
        if not name_adm1:
            continue

        # Remove '.' from name and replace ' ' by '_'
        name_adm0 = name_adm0.replace('.', '')
        name_adm1 = name_adm1.replace(' ', '_')

        if name_adm0 == 'uk_of_great_britain_and_northern_ireland':
            name_adm0 = 'u.k._of_great_britain_and_northern_ireland'

        # Create output directory
        dir_crop = get_crop_name(path_crop, parser.getboolean(name_adm0, 'USE_CROPLAND_MASK'))
        if not dir_crop:
            continue
        dir_out = dir_crop_masks / name_adm0 / dir_crop
        if use_adm2:
            path_out_ras = dir_out / f'{name_adm1}_{name_adm2}_{str(str_ID).zfill(9)}_{dir_crop}_crop_mask.tif'
        else:
            path_out_ras = dir_out / f'{name_adm1}_{str(str_ID).zfill(9)}_{dir_crop}_crop_mask.tif'

        # If subset file exists, then do not recreate it
        if not os.path.isfile(path_out_ras):
            if use_adm2:
                logger.info(f'{dir_out} {name_adm0} {name_adm1} {name_adm2}')
            else:
                logger.info(f'{dir_out} {name_adm0} {name_adm1}')
            utils.make_dir_if_missing(dir_out)

            # Open the global .tif file and split into ADM1s
            with rasterio.open(dat_level1) as src:
                b1 = src.read(1)

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
                pval = plot_hist(arr,
                                 dir_crop_masks / 'histograms' / name_adm0 / dir_crop,
                                 name_adm1,
                                 dir_crop,
                                 perc=constants.upper_percentile)

                df.loc[len(df)] = [name_adm0, name_adm1, dir_crop, pval/100.]
            except:
                logger.info('Cannot plot histogram: ' + name_adm0)

            try:
                with rasterio.open(path_out_ras, 'w', **profile) as dst:
                    dst.write(arr.astype(rasterio.int32), 1)
            except:
                if use_adm2:
                    logger.error('Cannot create crop-mask ' + name_adm0 + ' ' + name_adm1 + '_' + name_adm2 + '_' + str(str_ID).zfill(9) + '_' + dir_crop)
                else:
                    logger.error('Cannot create crop-mask ' + name_adm0 + ' ' + name_adm1 + '_' + str(str_ID).zfill(9) + '_' + dir_crop)

    return df


def run(params):
    crops = params.dir_masks.glob('*.tif')

    for country in params.countries:
        category = parser.get(country, 'category')

        df_cmask = gp.GeoDataFrame.from_file(params.dir_regions_shp / params.parser.get(country, 'shp_boundary'))
        df_cmask.fillna({'ADM0_NAME': '', 'ADM1_NAME': ''}, inplace=True)
        df_cmask['lcountry'] = df_cmask['country'].str.replace(' ', '_').str.lower()
        df_cmask = df_cmask[['ADM1_NAME', 'ADM0_NAME', 'Country_ID', 'Region_ID', 'num_ID', 'str_ID', 'R_ID', 'C_ID', 'lcountry']]
        pdb.set_trace()
        frames = []
        for crop in tqdm(crops):
            df = create_crop_masks(crop, country, df_cmask)
            df.loc[:, 'category'] = category

            frames.append(df)

    if len(frames):
        df_full = pd.concat(frames)
        df_full.to_csv(dir_crop_masks / 'stats_crop_masks.csv')


if __name__ == '__main__':
    # loop_get_crop_area()
    run()
