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

        self.countries = ast.literal_eval(self.parser.get('DEFAULT', 'countries'))

    def country_information(self, country, crop, scale):
        """
        Get country specific information.
        Args:
            country ():
            crop ():
            scale ():

        Returns:

        """
        self.country = country
        self.crop = crop
        self.scale = scale

        self.static_columns = ['country', 'region', 'region_id', 'year', 'doy']
        self.threshold = self.parser.getboolean(country, 'threshold')  # use threshold or percentile for crop masking
        limit_type = 'floor' if self.threshold else 'ceil'
        self.eo_model = ast.literal_eval(self.parser.get(country, 'eo_model'))  # list of EO variables to use in the model
        self.limit = self.parser.getint(country, limit_type)
        self.dir_threshold = f'crop_t{self.limit}' if self.threshold else f'crop_p{self.limit}'
        self.use_cropland_mask = self.parser.get(country, 'use_cropland_mask')
        self.seasons = ast.literal_eval(self.parser.get(country, 'seasons'))

        # dataset containing all data for a given country x crop x scale combination
        self.df_ccs = pd.DataFrame()

    def pretty_print(self, info='country_information'):
        """
        Args:
            info ():

        Returns:

        """
        self.logger.info('###################################################################')
        if info == 'country_information':
            self.logger.info(self.country)
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

    def merge_eo_files(self):
        """

        Args:

        Returns:

        """
        frames = []

        # For each element in vars, create a list of files to be merged
        for var in self.eo_model:
            path_var_files = self.dir_input / self.dir_threshold / self.country / self.scale / self.crop / var
            var_files = list(path_var_files.rglob('*.csv'))

            for fl in var_files:
                frames.append(pd.read_csv(fl, usecols=self.static_columns + [var]))

        # Merge files into dataframe
        df_result = None
        for df in tqdm(frames, total=len(frames), desc=f'Merging EO data', leave=False):
            if df_result is None:
                df_result = df.set_index(self.static_columns)
            else:
                df_result = df_result.combine_first(df.set_index(self.static_columns))

        # Reset index to get the right format
        df_result = df_result.reset_index()

        return df_result

    def add_static_information(self):
        """

        Args:

        Returns:

        """
        pos = len(self.static_columns)

        # Add static information
        self.df_ccs.insert(pos, 'crop', self.crop)
        self.df_ccs.insert(pos + 1, 'scale', self.scale)

        # Add datetime based on year and day of year
        self.df_ccs.insert(pos + 2, 'datetime', self.df_ccs.apply(lambda x: datetime.datetime.strptime(f'{x.year} {x.doy}', '%Y %j'), axis=1))

        # Add name of month, both abbreviated and full as well month number
        self.df_ccs.insert(pos + 3, 'abbr_month', self.df_ccs.apply(lambda x: x.datetime.strftime('%b'), axis=1))
        self.df_ccs.insert(pos + 4, 'name_month', self.df_ccs.apply(lambda x: x.datetime.strftime('%B'), axis=1))
        self.df_ccs.insert(pos + 5, 'Month', self.df_ccs.apply(lambda x: x.datetime.strftime('%m'), axis=1))


def run(path_config_file='geoextract.txt'):
    """

    Args:
        path_config_file ():

    Returns:

    """
    # Read in configuration file.
    gm = GeoMerge(path_config_file)
    gm.parse_config('DEFAULT')

    # Get all combinations of country, crop, scale to produce GEOCIF/AgMET inputs for
    all_combinations = gm.create_run_combinations()

    pbar = tqdm(all_combinations, total=len(all_combinations))
    for country, crop, scale in pbar:  # e.g. rwanda, cr, admin1
        pbar.set_description(f'Processing {country} {crop} {scale}')
        pbar.update()

        gm.country_information(country, crop, scale)
        gm.pretty_print(info='country_information')
        name_crop = 'cr' if gm.use_cropland_mask else crop

        dir_output = gm.dir_input / gm.dir_threshold / country / scale
        os.makedirs(dir_output, exist_ok=True)

        # create dataframe for country, crop and scale (ccs) by merging all EO files
        gm.df_ccs = gm.merge_eo_files()
        # and then adding static information like country, crop, scale, datetime, month, etc.
        gm.add_static_information()

        for season in gm.seasons:
            gm.logger.info(f'Storing output in {dir_output / f"eo_{country}_{scale}_{name_crop}_s{season}.csv"}')
            gm.df_ccs.to_csv(dir_output / f'eo_{country}_{scale}_{name_crop}_s{season}.csv', index=False)


if __name__ == '__main__':
    # Folder structure is as follows:
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<CROP>\
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<EO_DATA_FILE.csv>
    run()

