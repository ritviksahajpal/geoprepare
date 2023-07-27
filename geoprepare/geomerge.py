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
import numpy as np
from tqdm import tqdm

import geopandas as gp

dg = gp.read_file(r"C:\Users\ritvik\Downloads\MERRA\WoSIS-OrganicCarbon.shp")
breakpoint()

from . import base
from . import utils


class GeoMerge(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section="DEFAULT"):
        """

        Args:
            section ():

        Returns:

        """
        super().parse_config(section="DEFAULT")

        self.countries = ast.literal_eval(self.parser.get("DEFAULT", "countries"))

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

        self.static_columns = ["country", "region", "region_id", "year", "doy"]
        self.eo_model = ast.literal_eval(
            self.parser.get(country, "eo_model")
        )  # list of EO variables to use in the model
        self.use_cropland_mask = self.parser.get(country, "use_cropland_mask")
        self.get_dirname(country)

    def pretty_print(self, info="country_information"):
        """
        Args:
            info ():

        Returns:

        """
        self.logger.info(
            "###################################################################"
        )
        if info == "country_information":
            self.logger.info(self.country)
            self.logger.info(f"Scale: {self.scale}")
            self.logger.info(f"Crop: {self.crop}")
            self.logger.info(f"growing_season: {self.growing_season}")
            self.logger.info(f"Threshold used for crop masking: {self.threshold}")
            self.logger.info(f"Approach used for crop masking: {self.limit}")
            self.logger.info(f"EO variables to be processed: {self.eo_model}")
            self.logger.info(
                f"Use cropland (True) or crop (False) mask: {self.use_cropland_mask}"
            )
        self.logger.info(
            "####################################################################"
        )

    def create_run_combinations(self):
        """
        Create combinations of run parameters.
        Returns:

        """
        all_combinations = []

        for country in self.countries:
            self.scales = ast.literal_eval(self.parser.get(country, "scales"))
            self.crops = ast.literal_eval(self.parser.get(country, "crops"))
            self.growing_seasons = ast.literal_eval(
                self.parser.get(country, "growing_seasons")
            )

            for scale in self.scales:
                for crop in self.crops:
                    for growing_season in self.growing_seasons:
                        all_combinations.extend(
                            list(
                                itertools.product(
                                    [country], [scale], [crop], [growing_season]
                                )
                            )
                        )

        return all_combinations

    def merge_eo_files(self):
        """

        Args:

        Returns:

        """
        frames = []

        # For each element in vars, create a list of files to be merged
        for var in self.eo_model:
            crop_folder_name = "cr" if self.use_cropland_mask else "cr"
            path_var_files = (
                self.dir_input
                / self.dir_threshold
                / self.country
                / self.scale
                / crop_folder_name
                / var
            )

            # Find all the files for a given EO variable for a given country x crop x scale combination
            var_files = list(path_var_files.rglob("*.csv"))

            # Create a dataframe for each file and append to a list
            for fl in var_files:
                frames.append(pd.read_csv(fl, usecols=self.static_columns + [var]))

        # Merge files into dataframe
        df_result = None
        pbar = tqdm(frames, total=len(frames), leave=False)
        for df in pbar:
            pbar.set_description(f"Merging {var} files")
            pbar.update()

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
        self.df_ccs.insert(
            0,
            "datetime",
            self.df_ccs.apply(
                lambda x: datetime.datetime.strptime(f"{x.year} {x.doy}", "%Y %j"),
                axis=1,
            ),
        )

        # Add day of month, name of month, both abbreviated and full as well month number
        self.df_ccs.insert(
            pos + 1,
            "day",
            self.df_ccs.apply(lambda x: x.datetime.strftime("%d"), axis=1),
        )
        self.df_ccs.insert(
            pos + 2,
            "abbr_month",
            self.df_ccs.apply(lambda x: x.datetime.strftime("%b"), axis=1),
        )
        self.df_ccs.insert(
            pos + 3,
            "name_month",
            self.df_ccs.apply(lambda x: x.datetime.strftime("%B"), axis=1),
        )
        self.df_ccs.insert(
            pos + 4,
            "month",
            self.df_ccs.apply(lambda x: x.datetime.strftime("%m"), axis=1),
        )

        # Add static information
        self.df_ccs.insert(pos + 5, "crop", self.crop)
        self.df_ccs.insert(pos + 6, "scale", self.scale)

    def fillna(self, group, df_combination, df_stats):
        """
        Fill missing values in the dataframe with the unique values in that column
        Args:
            group (): Groups defined by Region (admin_1/admin_2), year combination
            df_combination (): Unique combination of scale, calendar_region, category, growing_season
            df_stats (): Yield, area and production statistics

        Returns:

        """
        df_sub = df_stats[
            ["country", self.scale, "year", "crop", "yield", "area", "production"]
        ]

        group = pd.merge(
            group,
            df_combination,
            left_on=["region"],
            right_on=[self.scale],
            how="inner",
        )

        # Rename self.scale to region
        df_sub = df_sub.rename(columns={self.scale: "region"})

        group = pd.merge(
            group, df_sub, on=["country", "region", "year", "crop"], how="left"
        )

        # Remove self.scale column from group
        group = group.drop(columns=[self.scale])

        return group

    def add_statistics(self):
        """
        Add yield, area and production statistics to the dataframe
        Returns:

        """
        # Subset statstics dataframe for current growing_season and crop
        df_stats = self.df_statistics[
            (self.df_statistics["country"] == self.country)
            & (self.df_statistics["crop"] == self.crop)
            & (self.df_statistics["growing_season"] == self.growing_season)
        ]

        # select appropriate region column based on scale
        if self.scale == "admin_1":
            df_stats = df_stats[df_stats["admin_2"].isna()]
        else:
            df_stats = df_stats[~df_stats["admin_2"].isna()]

        # For each country, scale combination get the scale, calendar_region, category combination
        df_combination = df_stats[
            [self.scale, "calendar_region", "category", "growing_season"]
        ]
        df_combination = df_combination.drop_duplicates()

        # Fill in missing values
        groups = self.df_ccs.groupby(["region", "year"])
        frames = []
        for name, group in groups:
            df_group = self.fillna(group, df_combination, df_stats)
            frames.append(df_group)

        df = pd.concat(frames)

        return df

    def read_calendar(self, group, year):
        """
        Read the crop calendar for the current group and year
        Args:
            group (): Groups defined by Region (admin_1/admin_2), year combination
            year (): Year for which

        Returns:

        """
        # Get the calendar region for the current group
        calendar_region = group["calendar_region"].unique()[0]

        # Get the calendar information for the current calendar_region
        df_cal = self.df_calendar[
            (self.df_calendar["country"] == self.country)
            & (self.df_calendar["calendar_region"] == calendar_region)
        ]

        # Get the calendar information for the current year and convert to a daily list
        if not df_cal.empty:
            df_tmp = utils.get_cal_list(df_cal.loc[:, "jan_1":], year)

            if not df_tmp.empty:
                group["crop_calendar"] = df_tmp["col"].values

        return group

    def add_calendar(self):
        """

        Returns:

        """
        frames = []

        # Loop through each calendar_region and year combination
        groups = self.df_ccs.groupby(["region", "year"])
        for name, group in tqdm(
            groups, desc=f"Adding calendar information to {self.scale}", leave=False
        ):
            _, year = name
            df_group = self.read_calendar(group, year)
            frames.append(df_group)

        df = pd.concat(frames)

        return df

    def assign_harvest_season(self, group):
        START_CROP_STAGE = 1
        END_CROP_STAGE = 3

        # reset index
        group = group.reset_index(drop=True)

        # replace 4 by 0 in crop calendar, 4 represents out of season values
        group_calendar = group["crop_calendar"].values
        group_calendar[group_calendar == 4] = 0

        # If 0 at start or end then season = year
        if (group_calendar[0] == 0) or (group_calendar[-1] == 0):
            # Find first occurence of 1
            idx_start = np.where(group_calendar == START_CROP_STAGE)[0][0]
            # Find last occurence of 3
            idx_end = np.where(group_calendar == END_CROP_STAGE)[0][-1] + 1

            group.loc[idx_start:idx_end, "harvest_season"] = int(
                group["year"].unique()[0]
            )
        else:
            # Find last occurence of 0
            idx_start = np.where(group_calendar == 0)[0][-1] + 1
            # Find first occ of 0
            idx_end = np.where(group_calendar == 0)[0][0] - 1

            group.loc[0:idx_end, "harvest_season"] = int(group["year"].unique()[0])

            if group["hemisphere"].unique()[0] == "N" and self.crop == "ww":
                group.loc[idx_start:365, "harvest_season"] = np.nan
            else:
                group.loc[idx_start:365, "harvest_season"] = int(
                    group["year"].unique()[0] + 1
                )

        return group

    def post_process(self):
        # 1. Scale NDVI
        self.df_ccs.loc[:, "ndvi"] = (self.df_ccs["ndvi"] - 50.0) / 200.0

        # 2. Assign hemisphere and temperate/tropical zones
        self.df_ccs = pd.merge(self.df_ccs, self.df_countries, on="country", how="left")

        # 3. Add average_temperature
        self.df_ccs["average_temperature"] = (
            self.df_ccs["cpc_tmax"] + self.df_ccs["cpc_tmin"]
        ) / 2.0

        # 4. Add harvest season information
        groups = self.df_ccs.groupby(["region", "year"])

        frames = []
        for name, group in groups:
            df_group = self.assign_harvest_season(group)
            frames.append(df_group)

        self.df_ccs = pd.concat(frames)
        self.df_ccs = self.df_ccs.reset_index(drop=True)

        # TODO: Add a dictionary for the growing season so that we can accomodate multiple types of growth stages
        # 5. Set growing_season to np.NaN when crop_calendar is 1, 2 or 3
        self.df_ccs.loc[
            ~self.df_ccs["crop_calendar"].isin([1, 2, 3]), "growing_season"
        ] = np.nan

        # 6. Add dekad information
        self.df_ccs["dekad"] = self.df_ccs["datetime"].dt.dayofyear // 10 + 1

        # 7. Move EO columns to the end of the dataframe, to make it more readable
        self.move_columns_to_end(columns=self.eo_model)

    def move_columns_to_end(self, columns=None):
        # Exclude elements from columns that are not in the dataframe
        columns = [x for x in columns if x in self.df_ccs.columns]

        # Move the columns to the end of the dataframe
        self.df_ccs = self.df_ccs[
            [x for x in self.df_ccs.columns if not x in columns] + columns
        ]


def run(path_config_file="geoextract.txt"):
    """

    Args:
        path_config_file ():

    Returns:

    """
    # Read in configuration file.
    gm = GeoMerge(path_config_file)
    gm.parse_config("DEFAULT")

    # Get all combinations of country, crop, scale, growing_season to produce GEOCIF/AgMET inputs for
    all_combinations = gm.create_run_combinations()

    # Read calendar and crop statistics
    gm.read_statistics(read_all=True)

    pbar = tqdm(all_combinations, total=len(all_combinations))
    for country, scale, crop, growing_season in pbar:  # e.g. rwanda, cr, admin1
        pbar.set_description(
            f"Crop: {crop} Growing season: {growing_season} Scale: {scale} {country.title()}"
        )
        pbar.update()

        # 1. Initialize GeoMerge object with country, scale, crop, growing_season
        gm.country_information(country, scale, crop, growing_season)
        gm.pretty_print(info="country_information")

        # 2. Set up output directory and file that stores the output
        dir_output = gm.dir_input / gm.dir_threshold / country / scale
        os.makedirs(dir_output, exist_ok=True)
        output_file = dir_output / f"{crop}_s{growing_season}.csv"

        # 3a. Check if crop calendar information exists for country, crop and growing_season
        # if empty then skip
        df_cal = gm.df_calendar[
            (gm.df_calendar["country"] == country)
            & (gm.df_calendar["crop"] == crop)
            & (gm.df_calendar["growing_season"] == growing_season)
        ]

        if not df_cal.empty:
            # 3b. Merge all EO data for country, crop and scale (ccs) into a dataframe
            gm.df_ccs = gm.merge_eo_files()

            # 4. Add static data: country, crop, scale, datetime, month, etc.
            gm.add_static_information()

            # 5. Fill in missing values, append additional blank year
            gm.df_ccs = utils.fill_missing_values(gm.df_ccs, gm.eo_model)

            # 5. Add yield, area, production information
            gm.df_ccs = gm.add_statistics()

            # Remove leap year day (Feb 29th) to make assigning calendar information easier
            # gm.df_ccs = utils.remove_leap_doy(gm.df_ccs)

            # 6. Add crop calendar information
            gm.df_ccs = gm.add_calendar()

            # 7. Post process data
            gm.post_process()

            # 8. Store output to disk
            if not gm.df_ccs.empty:
                gm.logger.info(f"Storing output in {output_file}")
                gm.df_ccs.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Folder structure is as follows:
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<CROP>\
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<EO_DATA_FILE.csv>
    run()
