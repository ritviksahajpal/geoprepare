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
from pathlib import Path
from tqdm import tqdm

import geopandas as gp

from . import base
from . import utils
from . import georegion
from . import log


class GeoMerge(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section="DEFAULT"):
        """

        Args:
            section ():

        Returns:

        """
        self.project_name = self.parser.get("PROJECT", "project_name")
        self.parallel_merge = self.parser.getboolean("PROJECT", "parallel_merge")
        super().parse_config(project_name=self.project_name, section="DEFAULT")

        self.countries = ast.literal_eval(self.parser.get("DEFAULT", "countries"))
        self.dir_boundary_files = Path(self.parser.get("PATHS", "dir_boundary_files"))

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
        self.use_cropland_mask = self.parser.getboolean(country, "use_cropland_mask")
        self.get_dirname(country)

    def pretty_print(self, info="country_information"):
        """
        Args:
            info ():

        Returns:

        """
        self.logger.info("###################################################")
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
        self.logger.info("###################################################")

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
        Merge all per-region/year EO CSV files into a single DataFrame.
        For each EO variable, concatenates all CSV files, then merges across
        variables on the static columns (country, region, region_id, year, doy).
        """
        df_result = None

        for var in self.eo_model:
            crop_folder_name = "cr" if self.use_cropland_mask else self.crop
            path_var_files = (
                self.dir_output
                / self.dir_threshold
                / self.country
                / self.scale
                / crop_folder_name
                / var
            )

            var_files = list(path_var_files.rglob("*.csv"))
            if not var_files:
                continue

            # Read and concat all files for this variable in one shot
            var_frames = [
                pd.read_csv(fl, usecols=self.static_columns + [var])
                for fl in var_files
            ]

            if not var_frames:
                continue

            df_var = pd.concat(var_frames, ignore_index=True)

            # Drop duplicate rows (same region/year/doy), keeping first non-NaN
            df_var = (
                df_var.groupby(self.static_columns, as_index=False)
                .first()
            )

            # Merge into result
            if df_result is None:
                df_result = df_var
            else:
                df_result = pd.merge(
                    df_result, df_var,
                    on=self.static_columns,
                    how="outer",
                )

        if df_result is None:
            df_result = pd.DataFrame(columns=self.static_columns)

        # Harmonize by converting to lower case and replacing space by _
        df_result = utils.harmonize_df(df_result)

        return df_result

    def add_static_information(self):
        """

        Args:

        Returns:

        """
        self.logger.info("Adding static information to the dataframe")

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
        group_calendar = group["crop_calendar"].values.copy()
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
        elif not np.any(group_calendar == 0):
            # No off-season at all — entire year is in-season
            group.loc[:, "harvest_season"] = int(group["year"].unique()[0])
        else:
            # Season wraps across year boundary — 0s are in the middle
            # Find last occurence of 0
            idx_start = np.where(group_calendar == 0)[0][-1] + 1
            # Find first occ of 0
            idx_end = np.where(group_calendar == 0)[0][0] - 1

            group.loc[0:idx_end, "harvest_season"] = int(group["year"].unique()[0])

            if group["hemisphere"].unique()[0] == "N" and self.crop in ("ww", "winter_wheat"):
                group.loc[idx_start:365, "harvest_season"] = np.nan
            else:
                group.loc[idx_start:365, "harvest_season"] = int(
                    group["year"].unique()[0] + 1
                )

        return group

    def post_process(self):
        # 1. Scale NDVI
        if "ndvi" in self.df_ccs.columns:
            self.df_ccs.loc[:, "ndvi"] = (self.df_ccs["ndvi"] - 50.0) / 200.0

        # 2. Assign hemisphere and temperate/tropical zones
        # Rename 'region' in df_countries to 'zone' to avoid clash with admin 'region'
        df_countries = self.df_countries.copy()
        if "region" in df_countries.columns:
            df_countries.rename(columns={"region": "zone"}, inplace=True)
        self.df_ccs = pd.merge(self.df_ccs, df_countries, on="country", how="left")

        # 3. Add average_temperature
        if "cpc_tmax" in self.df_ccs.columns and "cpc_tmin" in self.df_ccs.columns:
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
        # 5. Set growing_season to np.nan when crop_calendar is 1, 2 or 3
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


def process_combination(combination, path_config_file, parallel=False):
    """
    Worker function for a single combination of (country, scale, crop, growing_season).
    Returns (combination, success_boolean) for reporting.
    """
    country, scale, crop, growing_season = combination

    # Create a new GeoMerge instance for each combination
    gm = GeoMerge(path_config_file)
    gm.parse_config("DEFAULT")

    # In parallel mode, swap to a process-safe logger to avoid
    # file contention on the shared logzero RotatingFileHandler
    if parallel:
        gm.logger = log.SafeLogger(
            name=f"geoprepare.merge.{country}.{crop}",
            level=gm.compute_logging_level(),
        )

    # 1. Read statistics (you can pass read_all=True if needed)
    gm.read_statistics(country, crop, growing_season, read_all=True)

    # 2. Initialize GeoMerge object with country, scale, crop, growing_season
    gm.country_information(country, scale, crop, growing_season)
    gm.pretty_print(info="country_information")

    # 3. Set up output directory and output file
    dir_output = gm.dir_output / gm.dir_threshold / country
    os.makedirs(dir_output, exist_ok=True)
    output_file = dir_output / f"{country}_{crop}_s{growing_season}.csv"

    # 4. Check if crop calendar info exists
    df_cal = gm.df_calendar[gm.df_calendar["country"] == country]
    if df_cal.empty:
        return (combination, False)  # No data, skip

    # 5. Merge all EO data
    gm.df_ccs = gm.merge_eo_files()

    # 6. Add static data, fill missing, add yield stats, etc.
    gm.add_static_information()
    gm.df_ccs = utils.fill_missing_values(gm.df_ccs, gm.eo_model)
    # gm.df_ccs = gm.add_statistics()

    # Assign calendar_region from spatial overlay of admin units → EWCM regions
    path_admin_shp = gm.dir_boundary_files / gm.parser.get(country, "shp_boundary")
    path_region_shp = gm.dir_boundary_files / gm.parser.get(country, "shp_region")
    region_lookup = georegion.get_region_lookup(
        path_admin_shp=path_admin_shp,
        path_region_shp=path_region_shp,
        country=country,
        scale=scale,
        dir_cache=gm.dir_intermed / "region_cache",
    )
    gm.df_ccs["calendar_region"] = gm.df_ccs["region"].map(region_lookup)

    gm.df_ccs = gm.add_calendar()
    gm.post_process()

    # 7. Store output to disk if not empty
    if not gm.df_ccs.empty:
        gm.logger.info(f"Storing output in {output_file}")
        gm.df_ccs.to_csv(output_file, index=False)
        return (combination, True)
    else:
        return (combination, False)


def run(path_config_file=["geobase.txt", "geoextract.txt"]):
    """
    Main function that can run either sequentially or in parallel,
    depending on the `parallel` argument.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 1. Instantiate GeoMerge just to parse the config and create run combos
    gm_master = GeoMerge(path_config_file)
    gm_master.parse_config("DEFAULT")

    # 2. Get all combinations
    all_combinations = gm_master.create_run_combinations()

    if not gm_master.parallel_merge:
        # -- SEQUENTIAL EXECUTION --
        results = []
        pbar = tqdm(all_combinations, total=len(all_combinations))
        for combo in pbar:
            pbar.set_description(f"Processing {combo}")
            pbar.update()

            combo_result, success = process_combination(combo, path_config_file)
            results.append((combo_result, success))
        succeeded = sum(s for _, s in results)
        gm_master.logger.info(f"[SEQUENTIAL] {succeeded}/{len(results)} combinations processed successfully.")
    else:
        # -- PARALLEL EXECUTION --
        results = []
        num_cpus = int(gm_master.fraction_cpus * os.cpu_count())
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            # Submit tasks
            future_to_combo = {
                executor.submit(process_combination, combo, path_config_file, True): combo
                for combo in all_combinations
            }

            # Use tqdm to track progress of tasks as they complete
            for future in tqdm(
                as_completed(future_to_combo),
                total=len(future_to_combo),
                desc="Processing combos (parallel)",
                leave=True
            ):
                combo = future_to_combo[future]
                try:
                    combo_result, success = future.result()
                    results.append((combo_result, success))
                except Exception as e:
                    print(f"[ERROR] Combination {combo}: {e}")
                    results.append((combo, False))

        succeeded = sum(s for _, s in results)
        gm_master.logger.info(f"[PARALLEL] {succeeded}/{len(results)} combinations processed successfully.")


if __name__ == "__main__":
    # Folder structure is as follows:
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<CROP>\
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<EO_DATA_FILE.csv>
    run()