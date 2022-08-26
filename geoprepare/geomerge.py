###############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
# July 4, 2022
###############################################################################
import os
import ast
import pdb
import datetime
import itertools
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from . import base


class GeoMerge(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section='DEFAULT'):
        """

        Args:
            section ():

        Returns:

        """
        super().parse_config(section='DEFAULT')

        self.dir_models = Path(self.parser.get('PATHS', 'dir_models'))
        self.dir_model_inputs = self.dir_models / 'inputs'
        self.dir_model_outputs = self.dir_models / 'outputs'
        self.countries = ast.literal_eval(self.parser.get('DEFAULT', 'countries'))

    def get_country_information(self, country):
        """
        Get country information.
        Args:
            country ():

        Returns:

        """
        self.country = country

        # Get crop calendar information
        self.path_calendar = self.dir_input / 'crop_calendars' / self.parser.get('DEFAULT', 'calendar_file')
        self.df_calendar = pd.ExcelFile(self.path_calendar)

        # Get yield, area and production information
        self.path_stats = self.dir_input / 'statistics' / self.parser.get('DEFAULT', 'statistics_file')
        self.df_stats = pd.read_csv(self.path_stats)

        self.threshold = self.parser.getboolean(country, 'threshold')
        self.eo_model = ast.literal_eval(self.parser.get(country, 'eo_model'))
        limit_type = 'floor' if self.threshold else 'ceil'
        self.limit = self.parser.getint(country, limit_type)
        self.use_cropland_mask = self.parser.get(country, 'use_cropland_mask')
        self.seasons = ast.literal_eval(self.parser.get(country, 'season'))

        # dataset containing all data for a given country x crop x scale combination
        self.df_ccs = pd.DataFrame()

    def pp_country_information(self):
        self.logger.info('###################################################################')
        self.logger.info(self.country)
        self.logger.info(f'Path to crop calendar: {self.path_calendar}')
        self.logger.info(f'Path to Ag statistics: {self.path_stats}')
        self.logger.info(f'Threshold used for crop masking: {self.threshold}')
        self.logger.info(f'Approach used for crop masking: {self.limit}')
        self.logger.info(f'EO variables to be processed: {self.eo_model}')
        self.logger.info(f'Seasons: {self.seasons}')
        self.logger.info(f'Use cropland (True) or crop (False) mask: {self.use_cropland_mask}')
        self.logger.info('####################################################################')

    def create_run_combinations(self):
        """
        Create combinations of run parameters.
        Returns:

        """
        all_combinations = []

        for country in self.countries:
            scales = ast.literal_eval(self.parser.get(country, 'scale'))
            use_cropland_mask = self.parser.get(country, 'use_cropland_mask')
            crops = ['cr'] if use_cropland_mask else ast.literal_eval(self.parser.get(country, 'crops'))

            for crop in crops:
                for scale in scales:
                    all_combinations.extend(list(itertools.product([country], [crop], [scale])))

        return all_combinations

    def merge_eo_files(self, country, name_crop, scale):
        """

        Args:
            params ():
            country ():
            name_crop ():
            scale ():

        Returns:

        """
        frames = []
        cols = ['country', 'region', 'region_id', 'year', 'doy']

        dir_crop_inputs = Path(f'crop_t{self.limit}') if self.threshold else Path(f'crop_p{self.limit}')

        # For each element in vars, create a list of files to be merged
        for var in self.eo_model:
            path_var_files = self.dir_input / dir_crop_inputs / country / scale / name_crop / var
            var_files = list(path_var_files.rglob('*.csv'))

            for fl in var_files:
                frames.append(pd.read_csv(fl, usecols=cols + [var]))

        # Merge files into dataframe
        df_result = None
        for df in tqdm(frames, total=len(frames), desc=f'merging EO data', leave=False):
            if df_result is None:
                df_result = df.set_index(cols)
            else:
                df_result = df_result.combine_first(df.set_index(cols))

        return df_result


def run(path_config_file='geoextract.txt'):
    """

    Args:
        path_config_file ():

    Returns:

    """
    # Read in configuration file.
    gm = GeoMerge(path_config_file)
    gm.parse_config('DEFAULT')

    all_combinations = gm.create_run_combinations()

    pbar = tqdm(all_combinations)
    for country, crop, scale in pbar:
        pbar.set_description(f'Processing {country} {crop} {scale}')
        pbar.update()

        gm.get_country_information(country)
        gm.pp_country_information()
        name_crop = 'cr' if gm.use_cropland_mask else crop

        # create dataframe for country, crop and scale (ccs)
        gm.df_ccs = gm.merge_eo_files(country, name_crop, scale)
        gm.df_ccs.loc[:, 'scale'] = scale

        for season in gm.seasons:
            dir_output = gm.dir_model_inputs / country / scale
            os.makedirs(dir_output, exist_ok=True)

            gm.df_ccs.to_csv(dir_output / f'eo_{country}_{scale}_{name_crop}_s{season}.csv')


if __name__ == '__main__':
    run()
