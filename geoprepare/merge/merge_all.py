import os
import ast

import itertools
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd

from .. import common


def create_run_combinations(params):
    all_combinations = []

    for country in params.countries:
        crops = ast.literal_eval(params.parser.get(country, 'crops'))
        scales = ast.literal_eval(params.parser.get(country, 'scale'))

        for crop in crops:
            for scale in scales:
                all_combinations.extend(list(itertools.product([country], [crop], [scale])))

    return all_combinations


def merge_eo_files(params, country, crop, scale):
    frames = []
    cols = ['country', 'region', 'region_id', 'year', 'doy']

    use_cropland_mask = params.parser.get(country, 'use_cropland_mask')
    threshold = params.parser.getboolean(country, 'threshold')
    vars = ast.literal_eval(params.parser.get(country, 'eo_model'))
    limit = common.crop_mask_limit(params, country, threshold)

    name_crop = 'cr' if use_cropland_mask else crop
    dir_crop_inputs = Path(f'crop_t{limit}') if threshold else Path(f'crop_p{limit}')

    # For each element in vars, create a list of files to be merged
    for var in vars:
        path_var_files = params.dir_input / dir_crop_inputs / country / scale / name_crop / var
        var_files = list(path_var_files.rglob('*.csv'))

        for fl in var_files:
            frames.append(pd.read_csv(fl, usecols=cols + [var]))

    # Merge files into dataframe
    df_result = None
    for df in tqdm(frames, total=len(frames), desc='merging EO data', leave=False):
        if df_result is None:
            df_result = df.set_index(cols)
        else:
            df_result = df_result.combine_first(df.set_index(cols))

    return df_result


def run(params):
    num_cpus = int(params.fraction_cpus * cpu_count()) if params.parallel_process else 1

    all_combinations = create_run_combinations(params)

    pbar = tqdm(all_combinations)
    for country, crop, scale in pbar:
        # create dataframe for country, crop and scale (ccs)
        df_ccs = merge_eo_files(params, country, crop, scale)

        # Add crop calendar information

        # Add yield, area and production information

        # Merge all csv files
        breakpoint()


if __name__ == '__run__':
    pass
