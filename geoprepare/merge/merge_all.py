import os
import ast

from pathlib import Path
from multiprocessing import Pool, cpu_count

from . import common


def run(params):
    num_cpus = int(params.fraction_cpus * cpu_count()) if params.parallel_process else 1
    years = list(range(params.start_year, params.end_year + 1))

    for country in params.countries:
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
        path_eo_files = params.dir_input / dir_crop_inputs / country / scale / crop
        eo_files = list(path_eo_files.rglob('*.csv'))

        breakpoint()


if __name__ == '__run__':
    pass
