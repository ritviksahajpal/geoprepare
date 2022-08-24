import os
import ast

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd

from .. import common


def run(params):
    num_cpus = int(params.fraction_cpus * cpu_count()) if params.parallel_process else 1
    years = list(range(params.start_year, params.end_year + 1))
    cols = ['country', 'region', 'region_id', 'year', 'doy']

    pbar = tqdm(params.countries)
    for country in pbar:
        pbar.set_description(f'{country}')
        pbar.update()

        # Check if we use a cropland mask or not
        use_cropland_mask = params.parser.get(country, 'use_cropland_mask')
        crops = ast.literal_eval(params.parser.get(country, 'crops'))
        vars = ast.literal_eval(params.parser.get(country, 'eo_model'))
        scale = ast.literal_eval(params.parser.get(country, 'scale'))
        season = ast.literal_eval(params.parser.get(country, 'season'))
        threshold = params.parser.getboolean(country, 'threshold')
        limit = common.crop_mask_limit(params, country, threshold)

        for crop in crops:
            name_crop = 'cr' if use_cropland_mask else crop

        dir_crop_inputs = Path(f'crop_t{limit}') if threshold else Path(f'crop_p{limit}')

        # For each element in vars, create a list of files to be merged
        frames = []
        pb = tqdm(vars, leave=False)
        for var in pb:
            pb.set_description(f'Processing {var} for {country}')
            pb.update()

            path_var_files = params.dir_input / dir_crop_inputs / country / scale[0] / 'cr' / var
            var_files = list(path_var_files.rglob('*.csv'))

            for fl in var_files:
                frames.append(pd.read_csv(fl, usecols=cols + [var]))

        result = None
        for ix, df in tqdm(enumerate(frames), total=len(frames), desc='merging EO data'):
            if ix >= 19:
                breakpoint()
            if result is None:
                result = df.set_index(cols)
            else:
                result = result.combine_first(df.set_index(cols))


        # Merge all csv files
        breakpoint()


if __name__ == '__run__':
    pass
