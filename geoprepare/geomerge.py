###############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
# July 4, 2022
###############################################################################
import os
import ast
import datetime
import itertools
import pandas as pd

from tqdm import tqdm

from . import base
from . import utils


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

    def country_information(self, country, scale, crop, growing_season):
        """
        Get country specific information.
        Args:
            country ():
            scale ():
            crop ():
            growing_season ():

        Returns:

        """
        self.country = country
        self.crop = crop
        self.scale = scale
        self.growing_season = growing_season

        self.static_columns = ['country', 'region', 'region_id', 'year', 'doy']
        self.threshold = self.parser.getboolean(country, 'threshold')  # use threshold or percentile for crop masking
        limit_type = 'floor' if self.threshold else 'ceil'
        self.eo_model = ast.literal_eval(self.parser.get(country, 'eo_model'))  # list of EO variables to use in the model
        self.limit = self.parser.getint(country, limit_type)
        self.dir_threshold = f'crop_t{self.limit}' if self.threshold else f'crop_p{self.limit}'
        self.use_cropland_mask = self.parser.get(country, 'use_cropland_mask')

        # dataframe containing all data for a given country x crop x scale (ccs) combination
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
            self.logger.info(f'Scale: {self.scale}')
            self.logger.info(f'Crop: {self.crop}')
            self.logger.info(f'growing_season: {self.growing_season}')
            self.logger.info(f'Threshold used for crop masking: {self.threshold}')
            self.logger.info(f'Approach used for crop masking: {self.limit}')
            self.logger.info(f'EO variables to be processed: {self.eo_model}')
            self.logger.info(f'Use cropland (True) or crop (False) mask: {self.use_cropland_mask}')
        self.logger.info('####################################################################')

    def create_run_combinations(self):
        """
        Create combinations of run parameters.
        Returns:

        """
        all_combinations = []

        for country in self.countries:
            self.scales = ast.literal_eval(self.parser.get(country, 'scales'))
            self.crops = ast.literal_eval(self.parser.get(country, 'crops'))
            self.growing_seasons = ast.literal_eval(self.parser.get(country, 'growing_seasons'))

            for scale in self.scales:
                for crop in self.crops:
                    for growing_season in self.growing_seasons:
                        all_combinations.extend(list(itertools.product([country], [scale], [crop], [growing_season])))

        return all_combinations

    def merge_eo_files(self):
        """

        Args:

        Returns:

        """
        frames = []

        # For each element in vars, create a list of files to be merged
        for var in self.eo_model:
            crop_folder_name = 'cr' if self.use_cropland_mask else 'cr'
            path_var_files = self.dir_input / self.dir_threshold / self.country / self.scale / crop_folder_name / var

            # Find all the files for a given EO variable for a given country x crop x scale combination
            var_files = list(path_var_files.rglob('*.csv'))

            # Create a dataframe for each file and append to a list
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

        # Harmonize by converting to lower case and replacing space by _
        df_result = utils.harmonize_df(df_result)

        return df_result

    def add_static_information(self):
        """

        Args:

        Returns:

        """
        pos = len(self.static_columns)

        # Add datetime based on year and day of year, make this the first column
        self.df_ccs.insert(0, 'datetime', self.df_ccs.apply(lambda x: datetime.datetime.strptime(f'{x.year} {x.doy}', '%Y %j'), axis=1))

        # Add name of month, both abbreviated and full as well month number
        self.df_ccs.insert(pos + 1, 'abbr_month', self.df_ccs.apply(lambda x: x.datetime.strftime('%b'), axis=1))
        self.df_ccs.insert(pos + 2, 'name_month', self.df_ccs.apply(lambda x: x.datetime.strftime('%B'), axis=1))
        self.df_ccs.insert(pos + 3, 'Month', self.df_ccs.apply(lambda x: x.datetime.strftime('%m'), axis=1))

        # Add static information
        self.df_ccs.insert(pos + 4, 'crop', self.crop)
        self.df_ccs.insert(pos + 5, 'scale', self.scale)

    def fillna(self, group, df_combination, df_stats):
        df_sub = df_stats[['country', self.scale, 'year', 'crop', 'yield', 'area', 'production']]

        group = pd.merge(group, df_combination, left_on=['region'], right_on=[self.scale], how='inner')

        # Rename self.scale to region
        df_sub = df_sub.rename(columns={self.scale: 'region'})

        group = pd.merge(group, df_sub, on=['country', 'region', 'year', 'crop'], how='left')

        # Remove self.scale column from group
        group = group.drop(columns=[self.scale])

        # Move the columns: calendar_region  category  growing_season  yield  area  production to len(self.static_columns) + 3
        cols = group.columns.tolist()
        cols = cols[:len(self.static_columns) + 3] + cols[-3:] + cols[len(self.static_columns) + 3:-3]
        group = group[cols]

        return group

    def read_statistics(self):
        """

        Args:

        Returns:

        """
        # Get crop calendar information
        self.path_calendar = self.dir_input / 'crop_calendars' / self.parser.get('DEFAULT', 'calendar_file')
        self.df_calendar = pd.read_csv(self.path_calendar) if os.path.isfile(self.path_calendar) else pd.DataFrame()
        self.df_calendar = utils.harmonize_df(self.df_calendar)

        # Get yield, area and production information
        self.path_stats = self.dir_input / 'statistics' / self.parser.get('DEFAULT', 'statistics_file')
        self.df_stats = pd.read_csv(self.path_stats) if os.path.isfile(self.path_stats) else pd.DataFrame()
        self.df_stats = utils.harmonize_df(self.df_stats)

    def add_statistics(self):
        """

        Returns:

        """
        # Subset statstics dataframe for current growing_season and crop
        df_stats = self.df_stats[(self.df_stats['country'] == self.country) &
                                 (self.df_stats['crop'] == self.crop) &
                                 (self.df_stats['growing_season'] == self.growing_season)]

        # select appropriate region column based on scale
        if self.scale == 'admin_1':
            df_stats = df_stats[df_stats['admin_2'].isna()]
        else:
            df_stats = df_stats[~df_stats['admin_2'].isna()]

        # For each country, scale combination get the scale, calendar_region, category combination
        df_combination = df_stats[[self.scale, 'calendar_region', 'category', 'growing_season']]
        df_combination = df_combination.drop_duplicates()

        groups = self.df_ccs.groupby(['region', 'year'])
        frames = []
        for name, group in groups:
            df_group = self.fillna(group, df_combination, df_stats)
            frames.append(df_group)

        df = pd.concat(frames)

        return df

    def read_calendar(self, group, year):
        # Get the calendar region for the current group
        calendar_region = group['calendar_region'].unique()[0]

        # Get the calendar information for the current calendar_region
        df_cal = self.df_calendar[(self.df_calendar['country'] == self.country) &
                                  (self.df_calendar['calendar_region'] == calendar_region)]

        # Get the calendar information for the current year and convert to a daily list
        if not df_cal.empty:
            df_tmp = utils.get_cal_list(df_cal.loc[:, 'jan_1':], year)

            if not df_tmp.empty:
                group['crop_calendar'] = df_tmp['col'].values

        return group

    def add_calendar(self):
        """

        Returns:

        """
        frames = []

        # Loop through each calendar_region and year combination
        groups = self.df_ccs.groupby(['region', 'year'])
        for name, group in tqdm(groups, desc=f'Adding calendar information to {self.scale}', leave=False):
            _, year = name
            df_group = self.read_calendar(group, year)
            frames.append(df_group)

        df = pd.concat(frames)

        return df


def run(path_config_file='geoextract.txt'):
    """

    Args:
        path_config_file ():

    Returns:

    """
    # Read in configuration file.
    gm = GeoMerge(path_config_file)
    gm.parse_config('DEFAULT')

    # Get all combinations of country, crop, scale, growing_season to produce GEOCIF/AgMET inputs for
    all_combinations = gm.create_run_combinations()

    # Read calendar and crop statistics
    gm.read_statistics()

    pbar = tqdm(all_combinations, total=len(all_combinations))
    for country, scale, crop, growing_season in pbar:  # e.g. rwanda, cr, admin1
        pbar.set_description(f'{country} Scale: {scale} Crop: {crop} Growing season: {growing_season}')
        pbar.update()

        # 1. Initialize GeoMerge object with country, scale, crop, growing_season
        gm.country_information(country, scale, crop, growing_season)
        gm.pretty_print(info='country_information')

        # 2. Set up output directory and file that stores the output
        dir_output = gm.dir_input / gm.dir_threshold / country / scale
        os.makedirs(dir_output, exist_ok=True)
        output_file = dir_output / f"{crop}_s{growing_season}.csv"

        # 3a. Check if crop calendar information exists for country, crop and scale, if not then bail
        df_cal = gm.df_calendar[(gm.df_calendar['country'] == gm.country) &
                                (gm.df_calendar['crop'] == gm.crop) &
                                (gm.df_calendar['scale'] == gm.scale)]

        if not df_cal.empty:
            # 3b. Merge all EO data for country, crop and scale (ccs) into a dataframe
            gm.df_ccs = gm.merge_eo_files()

            # 4. Add static data: country, crop, scale, datetime, month, etc.
            gm.add_static_information()

            # 5. Add yield, area, production information
            gm.df_ccs = gm.add_statistics()

            # 6. Add crop calendar information
            gm.df_ccs = gm.add_calendar()

            # 7. Store output to disk
            if not gm.df_ccs.empty:
                gm.logger.info(f'Storing output in {output_file}')
                gm.df_ccs.to_csv(output_file, index=False)


if __name__ == '__main__':
    # Folder structure is as follows:
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<CROP>\
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<EO_DATA_FILE.csv>
    run()

