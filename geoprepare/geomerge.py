"""
geomerge.py - Merge extracted EO statistics into per-country crop CSV files.

Combines per-region/year CSV files from the extract step into a single
DataFrame per country-crop-season combination. Adds crop calendar info,
harvest season assignment, and region-to-EWCM-region mapping.

Pipeline: extract (extract_EO.py) -> merge (geomerge.py) -> model (geocif)

Output: {dir_output}/crop_t{threshold}/{country}/{country}_{crop}_s{season}.csv
"""
import os
import gc
import ast
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
        self.project_name = self.parser.get("DEFAULT", "project_name")
        self.parallel_merge = self.parser.getboolean("DEFAULT", "parallel_merge")
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

        self.static_columns = ["country", "region", "region_id", "lat", "lon", "year", "doy"]
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
            self.scales = [self.parser.get(country, "admin_level")]
            self.crops = ast.literal_eval(self.parser.get(country, "crops"))
            self.growing_seasons = ast.literal_eval(
                self.parser.get(country, "seasons")
            )

            for scale in self.scales:
                for crop in self.crops:
                    crop_name = self.get_crop_full_name(crop)
                    for growing_season in self.growing_seasons:
                        # Wheat crops have a single season — skip season 2+
                        # to avoid producing a duplicate _s2.csv with
                        # identical data (calendar sheet is season-agnostic).
                        if crop_name in ("winter_wheat", "spring_wheat") and growing_season > 1:
                            self.logger.warning(
                                f"Skipping {crop_name} season {growing_season} "
                                f"for {country} — wheat has only 1 season"
                            )
                            continue
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

        AEF is handled specially: it has 64 columns (aef_1..aef_64) with no doy
        or year dimension (multi-year average), so it is merged on (country, region, region_id).
        """
        AEF_NUM_BANDS = 64
        df_result = None

        # Process daily vars first so df_result has year+doy rows;
        # AEF (no doy) and FLDAS (monthly) merge onto these rows afterward.
        FLDAS_NUM_LEADS = 6
        vars_ordered = sorted(
            self.eo_model,
            key=lambda v: (1 if v == "aef" else 2 if v.startswith("fldas_") or v.startswith("s2s_") else 0),
        )

        pbar = tqdm(vars_ordered, desc=f"Merging EO variables ({self.country})")
        for var in pbar:
            pbar.set_postfix_str(var)
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

            # AEF: 64-band average data (no doy or year column)
            if var == "aef":
                aef_cols = [f"aef_{i}" for i in range(1, AEF_NUM_BANDS + 1)]
                read_cols = ["country", "region", "region_id"] + aef_cols
                merge_cols = ["country", "region", "region_id"]
            elif var.startswith("fldas_"):
                # FLDAS: monthly data with 6 lead columns, merge on year+month
                fldas_lead_cols = [f"{var}_lead{i}" for i in range(FLDAS_NUM_LEADS)]
                read_cols = ["country", "region", "region_id", "year", "month"] + fldas_lead_cols
                merge_cols = None  # handled specially below
            elif var.startswith("s2s_"):
                # NOAA S2S: monthly seasonal forecasts with 6 lead columns
                S2S_NUM_LEADS = 6
                s2s_lead_cols = [f"{var}_lead{i}" for i in range(1, S2S_NUM_LEADS + 1)]
                read_cols = ["country", "region", "region_id", "year", "month"] + s2s_lead_cols
                merge_cols = None  # handled like FLDAS via year+month join
            else:
                read_cols = self.static_columns + [var]
                merge_cols = self.static_columns

            # Read and concat all files for this variable in one shot
            # Disable GC during bulk read to avoid progressive slowdown from
            # cyclic GC scanning 1000+ small DataFrames
            gc.disable()
            try:
                var_frames = [
                    pd.read_csv(fl, usecols=read_cols)
                    for fl in tqdm(var_files, desc=f"Reading {var} files", leave=False)
                ]
            finally:
                gc.enable()

            if not var_frames:
                continue

            df_var = pd.concat(var_frames, ignore_index=True)

            if var.startswith("fldas_") or var.startswith("s2s_"):
                # FLDAS / S2S: monthly data — broadcast to all DOYs via year+month join
                monthly_merge_cols = ["country", "region", "region_id", "year", "month"]
                df_var = df_var.groupby(monthly_merge_cols, as_index=False).first()

                if df_result is not None and "doy" in df_result.columns:
                    # Compute month from year+doy so we can join
                    df_result["_month"] = pd.to_datetime(
                        df_result["year"] * 1000 + df_result["doy"], format="%Y%j"
                    ).dt.month
                    df_result = pd.merge(
                        df_result, df_var,
                        left_on=["country", "region", "region_id", "year", "_month"],
                        right_on=monthly_merge_cols,
                        how="left",
                    )
                    df_result.drop(columns=["_month", "month"], inplace=True)
            else:
                # Drop duplicate rows, keeping first non-NaN
                df_var = (
                    df_var.groupby(merge_cols, as_index=False)
                    .first()
                )

                # Merge into result
                if df_result is None:
                    df_result = df_var
                else:
                    df_result = pd.merge(
                        df_result, df_var,
                        on=merge_cols,
                        how="outer" if var != "aef" else "left",
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

        # Add datetime based on year and day of year (vectorized)
        dt = pd.to_datetime(
            self.df_ccs["year"] * 1000 + self.df_ccs["doy"], format="%Y%j"
        )
        self.df_ccs.insert(0, "datetime", dt)

        # Add day/month columns (vectorized via .dt accessor)
        self.df_ccs.insert(pos + 1, "day", dt.dt.strftime("%d"))
        self.df_ccs.insert(pos + 2, "abbr_month", dt.dt.strftime("%b"))
        self.df_ccs.insert(pos + 3, "name_month", dt.dt.strftime("%B"))
        self.df_ccs.insert(pos + 4, "month", dt.dt.strftime("%m"))

        # Add static information
        self.df_ccs.insert(pos + 5, "crop", self.crop)
        self.df_ccs.insert(pos + 6, "scale", self.scale)
        self.df_ccs.insert(pos + 7, "growing_season", self.growing_season)

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
            (self.df_calendar["country2"] == self.country)
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

        # Defragment before groupby to avoid PerformanceWarning in the loop
        groups = self.df_ccs.copy().groupby(["region", "year"])
        for name, group in tqdm(
            groups, desc=f"Adding calendar information to {self.scale}", leave=False
        ):
            _, year = name
            df_group = self.read_calendar(group, year)
            frames.append(df_group)

        df = pd.concat(frames).copy()

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

            group.loc[idx_start:365, "harvest_season"] = int(
                group["year"].unique()[0] + 1
            )

        return group

    # Fallback pairs: (primary high-res column, coarser global fallback).
    # If both columns exist after merge, NaN in the primary is filled from
    # the fallback, then the fallback column is dropped — downstream code
    # only sees the primary name.  Activates automatically when the user
    # extracts both datasets in eo_model.
    _FALLBACK_PAIRS = [
        ("chirps", "cpc_precip"),
        ("chirts_era5_tmax", "cpc_tmax"),
        ("chirts_era5_tmin", "cpc_tmin"),
        ("daymet_tmax", "cpc_tmax"),
        ("daymet_tmin", "cpc_tmin"),
        ("daymet_prcp", "cpc_precip"),
    ]

    def _apply_fallback(self):
        """Fill NaN in primary EO columns from coarser fallback sources.

        Only activates when both primary and fallback columns exist in
        self.df_ccs (i.e. user listed both in ``eo_model``).  Drops the
        fallback column after fill so downstream code never sees the
        coarser source.
        """
        for primary, fallback in self._FALLBACK_PAIRS:
            if primary not in self.df_ccs.columns or fallback not in self.df_ccs.columns:
                continue
            n_before = int(self.df_ccs[primary].isna().sum())
            self.df_ccs[primary] = self.df_ccs[primary].fillna(self.df_ccs[fallback])
            n_filled = n_before - int(self.df_ccs[primary].isna().sum())
            if n_filled > 0:
                self.logger.info(
                    f"Fallback: filled {n_filled} NaN in '{primary}' from '{fallback}'"
                )
            self.df_ccs = self.df_ccs.drop(columns=[fallback])

    def post_process(self):
        # 1. Scale NDVI
        if "ndvi" in self.df_ccs.columns:
            self.df_ccs.loc[:, "ndvi"] = (self.df_ccs["ndvi"] - 50.0) / 200.0

        # 1b. Fill gaps in primary EO columns from coarser fallback sources
        #     (e.g. CHIRPS NaN above 60°N filled from CPC).  Must run before
        #     average_temperature computation so the filled values feed through.
        self._apply_fallback()

        # 2. Assign hemisphere and temperate/tropical zones
        # Rename 'region' in df_countries to 'zone' to avoid clash with admin 'region'
        df_countries = self.df_countries.copy()
        if "region" in df_countries.columns:
            df_countries.rename(columns={"region": "zone"}, inplace=True)
        self.df_ccs = pd.merge(self.df_ccs, df_countries, on="country", how="left")

        # 3. Add average_temperature
        tmax_col = tmin_col = None
        if "chirts_era5_tmax" in self.df_ccs.columns and "chirts_era5_tmin" in self.df_ccs.columns:
            tmax_col, tmin_col = "chirts_era5_tmax", "chirts_era5_tmin"
        elif "cpc_tmax" in self.df_ccs.columns and "cpc_tmin" in self.df_ccs.columns:
            tmax_col, tmin_col = "cpc_tmax", "cpc_tmin"

        if tmax_col and tmin_col:
            self.df_ccs["average_temperature"] = (
                self.df_ccs[tmax_col] + self.df_ccs[tmin_col]
            ) / 2.0

        # 4. Add harvest season information
        groups = self.df_ccs.groupby(["region", "year"])

        frames = []
        for name, group in groups:
            df_group = self.assign_harvest_season(group)
            frames.append(df_group)

        self.df_ccs = pd.concat(frames)
        self.df_ccs = self.df_ccs.reset_index(drop=True)

        # 5. Clear growing_season for off-season rows (crop_calendar not 1, 2, or 3)
        self.df_ccs.loc[
            ~self.df_ccs["crop_calendar"].isin([1, 2, 3]), "growing_season"
        ] = np.nan

        # 6. Add dekad information
        self.df_ccs["dekad"] = self.df_ccs["datetime"].dt.dayofyear // 10 + 1

        # 7. Move EO columns to the end of the dataframe, to make it more readable
        # Expand "aef" to actual column names aef_1..aef_64
        eo_columns = self._expand_eo_columns()
        self.move_columns_to_end(columns=eo_columns)

    def _expand_eo_columns(self):
        """Expand shorthand names in eo_model to actual column names.

        - 'aef' -> aef_1..aef_64
        - 'fldas_*' -> {var}_lead0..{var}_lead5
        - 's2s_*' -> {var}_lead1..{var}_lead6
        """
        FLDAS_NUM_LEADS = 6
        S2S_NUM_LEADS = 6
        expanded = []
        for var in self.eo_model:
            if var == "aef":
                expanded.extend([f"aef_{i}" for i in range(1, 65)])
            elif var.startswith("fldas_"):
                expanded.extend([f"{var}_lead{i}" for i in range(FLDAS_NUM_LEADS)])
            elif var.startswith("s2s_"):
                expanded.extend([f"{var}_lead{i}" for i in range(1, S2S_NUM_LEADS + 1)])
            else:
                expanded.append(var)
        return expanded

    def add_pre_season_rows(self, n_months):
        """
        Add synthetic monthly rows for N months before the first in-season month.

        These rows carry ONLY FLDAS/S2S lead columns (read from per-variable CSVs).
        All observational EO columns (ndvi, chirps, chirts, esi, etc.) are NaN.
        """
        if n_months <= 0 or self.df_ccs.empty:
            return

        FLDAS_NUM_LEADS = 6
        S2S_NUM_LEADS = 6

        # Identify monthly-lead variables in eo_model
        monthly_vars = [v for v in self.eo_model if v.startswith("fldas_") or v.startswith("s2s_")]
        if not monthly_vars:
            return

        # Build lead column names
        monthly_lead_cols = []
        for v in monthly_vars:
            if v.startswith("fldas_"):
                monthly_lead_cols.extend([f"{v}_lead{i}" for i in range(FLDAS_NUM_LEADS)])
            elif v.startswith("s2s_"):
                monthly_lead_cols.extend([f"{v}_lead{i}" for i in range(1, S2S_NUM_LEADS + 1)])

        # Read per-variable CSVs for monthly variables to get pre-season data
        crop_folder_name = "cr" if self.use_cropland_mask else self.crop
        monthly_data = {}  # {var: DataFrame with region_id, year, month, lead cols}
        for var in monthly_vars:
            path_var = (
                self.dir_output / self.dir_threshold / self.country
                / self.scale / crop_folder_name / var
            )
            var_files = list(path_var.rglob("*.csv")) if path_var.exists() else []
            if var_files:
                if var.startswith("fldas_"):
                    lead_cols = [f"{var}_lead{i}" for i in range(FLDAS_NUM_LEADS)]
                else:
                    lead_cols = [f"{var}_lead{i}" for i in range(1, S2S_NUM_LEADS + 1)]
                read_cols = ["region_id", "year", "month"] + lead_cols
                frames = []
                for f in var_files:
                    try:
                        df = pd.read_csv(f, usecols=lambda c: c in read_cols)
                        frames.append(df)
                    except Exception:
                        continue
                if frames:
                    df_concat = pd.concat(frames, ignore_index=True)
                    # Harmonize: lowercase column names and region_id to match merged data
                    df_concat.columns = [c.lower() for c in df_concat.columns]
                    if "region_id" in df_concat.columns:
                        df_concat["region_id"] = df_concat["region_id"].str.lower()
                    monthly_data[var] = df_concat

        if not monthly_data:
            self.logger.warning("No monthly lead data found for pre-season rows")
            return

        for var, df_m in monthly_data.items():
            self.logger.info(f"  Pre-season: loaded {len(df_m)} rows for {var} "
                             f"(cols: {list(df_m.columns)[:5]}...)")

        # Build set of existing (region, year, month) to avoid duplicate rows
        if "month" not in self.df_ccs.columns:
            self.df_ccs["month"] = pd.to_datetime(
                self.df_ccs["year"] * 1000 + self.df_ccs["doy"], format="%Y%j"
            ).dt.month
        existing_rym = set(
            zip(self.df_ccs["region"], self.df_ccs["year"], self.df_ccs["month"])
        )

        # For each (region, harvest_season), find the first in-season month
        pre_season_frames = []
        for (region, harvest_season), grp in self.df_ccs.groupby(["region", "harvest_season"]):
            if pd.isna(harvest_season):
                continue

            harvest_season = int(harvest_season)
            in_season = grp[grp["crop_calendar"].isin([1, 2, 3])]
            if in_season.empty:
                continue

            first_in_season_doy = int(in_season["doy"].min())
            first_in_season_month = pd.Timestamp(f"{harvest_season}-01-01") + pd.Timedelta(days=first_in_season_doy - 1)
            first_month = first_in_season_month.month

            # Get region metadata from existing data
            region_id = grp["region_id"].iloc[0]
            country = grp["country"].iloc[0]
            lat = grp["lat"].iloc[0] if "lat" in grp.columns else np.nan
            lon = grp["lon"].iloc[0] if "lon" in grp.columns else np.nan
            calendar_region = grp["calendar_region"].iloc[0] if "calendar_region" in grp.columns else np.nan

            # Compute pre-season months (going backwards, wrapping around year)
            pre_months = []
            for i in range(1, n_months + 1):
                m = first_month - i
                y = harvest_season
                if m <= 0:
                    m += 12
                    y -= 1
                doy_mid = pd.Timestamp(f"{y}-{m:02d}-15").timetuple().tm_yday
                pre_months.append((y, m, doy_mid))

            for y, m, doy_mid in pre_months:
                # Skip if daily data already exists for this region/year/month
                if (region, y, m) in existing_rym:
                    continue
                row = {
                    "country": country,
                    "region": region,
                    "region_id": region_id,
                    "year": y,
                    "doy": doy_mid,
                    "month": m,
                    "harvest_season": float(harvest_season),
                    "crop_calendar": 0.0,
                    "calendar_region": calendar_region,
                }
                if "lat" in self.df_ccs.columns:
                    row["lat"] = lat
                if "lon" in self.df_ccs.columns:
                    row["lon"] = lon

                pre_season_frames.append(row)

        if not pre_season_frames:
            return

        df_pre = pd.DataFrame(pre_season_frames)

        # Add datetime column
        df_pre["datetime"] = pd.to_datetime(
            df_pre["year"] * 1000 + df_pre["doy"], format="%Y%j"
        )

        # Join monthly lead data onto pre-season rows
        for var, df_monthly in monthly_data.items():
            merge_keys = ["region_id", "year", "month"]
            available_keys = [k for k in merge_keys if k in df_monthly.columns]
            if len(available_keys) == len(merge_keys):
                df_monthly_dedup = df_monthly.groupby(available_keys, as_index=False).first()
                df_pre = pd.merge(df_pre, df_monthly_dedup, on=available_keys, how="left")

        # Log join results
        for var in monthly_vars:
            if var.startswith("fldas_"):
                check_col = f"{var}_lead0"
            else:
                check_col = f"{var}_lead1"
            if check_col in df_pre.columns:
                n_filled = df_pre[check_col].notna().sum()
                self.logger.info(f"  Pre-season join: {check_col} has {n_filled}/{len(df_pre)} non-NaN values")

        # Ensure all EO columns exist (as NaN for observational vars)
        for col in self.df_ccs.columns:
            if col not in df_pre.columns:
                df_pre[col] = np.nan

        # Reorder columns to match main dataframe
        df_pre = df_pre.reindex(columns=self.df_ccs.columns)

        # Append and sort
        self.df_ccs = pd.concat([self.df_ccs, df_pre], ignore_index=True)
        self.df_ccs = self.df_ccs.sort_values(
            ["region", "year", "doy"]
        ).reset_index(drop=True)

        self.logger.info(
            f"Added {len(df_pre)} pre-season rows ({n_months} months before planting)"
        )

    def move_columns_to_end(self, columns=None):
        # Exclude elements from columns that are not in the dataframe
        columns = [x for x in columns if x in self.df_ccs.columns]

        # Move the columns to the end of the dataframe
        self.df_ccs = self.df_ccs[
            [x for x in self.df_ccs.columns if not x in columns] + columns
        ]


def process_combination(combination, path_config_file, parallel=False, df_eo=None):
    """
    Worker function for a single combination of (country, scale, crop, growing_season).
    Returns (combination, success_boolean) for reporting.

    If df_eo is provided, skip the expensive merge_eo_files() call and use the
    pre-read EO DataFrame instead.
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
    gm.read_statistics(country,
                       crop,
                       growing_season,
                       read_calendar=True,
                       read_statistics=False,
                       read_countries=True)

    # 2. Initialize GeoMerge object with country, scale, crop, growing_season
    gm.country_information(country, scale, crop, growing_season)
    gm.pretty_print(info="country_information")

    # 3. Set up output directory and output file
    dir_output = gm.dir_output / gm.dir_threshold / country
    os.makedirs(dir_output, exist_ok=True)
    output_file = dir_output / f"{country}_{crop}_s{growing_season}.csv"

    # 4. Check if crop calendar info exists
    if gm.df_calendar.empty or "country2" not in gm.df_calendar.columns:
        gm.logger.error(f"Skipping {combination}: calendar is empty or missing 'country2' column (path: {gm.path_calendar})")
        return (combination, False)
    df_cal = gm.df_calendar[gm.df_calendar["country2"] == country]
    if df_cal.empty:
        gm.logger.error(f"Skipping {combination}: no calendar rows for country '{country}'")
        return (combination, False)

    # 5. Merge all EO data (or reuse pre-read data)
    if df_eo is not None:
        gm.df_ccs = df_eo.copy()
    else:
        gm.df_ccs = gm.merge_eo_files()

    # 6. Add static data, fill missing, add yield stats, etc.
    gm.add_static_information()
    gm.df_ccs = utils.fill_missing_values(gm.df_ccs, gm._expand_eo_columns())

    # Assign calendar_region from spatial overlay of admin units → EWCM regions
    path_admin_shp = gm.dir_boundary_files / gm.parser.get(country, "boundary_file")
    path_region_shp = gm.dir_boundary_files / gm.parser.get(country, "shp_region")
    region_lookup = georegion.get_region_lookup(
        path_admin_shp=path_admin_shp,
        path_region_shp=path_region_shp,
        country=country,
        scale=scale,
        dir_cache=gm.dir_intermed / "region_cache",
    )
    gm.df_ccs["calendar_region"] = gm.df_ccs["region"].map(region_lookup)

    # Filter out regions that have no calendar data for this crop.
    # (e.g. poppy is only grown in 4 of myanmar's 17 admin regions)
    valid_regions = set(
        gm.df_calendar.loc[gm.df_calendar["country2"] == country, "calendar_region"].unique()
    )
    before_regions = set(gm.df_ccs["region"].unique())
    gm.df_ccs = gm.df_ccs[gm.df_ccs["calendar_region"].isin(valid_regions)]
    after_regions = set(gm.df_ccs["region"].unique())
    dropped = sorted(before_regions - after_regions)
    if dropped:
        gm.logger.warning(
            f"{country}/{crop}: dropped {len(dropped)} regions with no calendar data: "
            f"{', '.join(dropped[:10])}{'...' if len(dropped) > 10 else ''}"
        )

    if gm.df_ccs.empty:
        gm.logger.error(
            f"Skipping {combination}: no admin regions have calendar data for {crop}"
        )
        return (combination, False)

    gm.df_ccs = gm.add_calendar()
    gm.post_process()

    # 6b. Add pre-season rows with FLDAS/S2S lead columns
    pre_season_months = gm.parser.getint("DEFAULT", "pre_season_months", fallback=0)
    if pre_season_months > 0:
        gm.add_pre_season_rows(pre_season_months)

    # 7. Store output to disk if not empty
    if not gm.df_ccs.empty:
        gm.logger.info(f"Storing output in {output_file}")
        gm.df_ccs.to_csv(output_file, index=False)
        return (combination, True)
    else:
        gm.logger.error(f"Skipping {combination}: merged DataFrame is empty after processing")
        return (combination, False)


def process_group(group_key, group_combos, path_config_file, parallel=False):
    """
    Process a group of combinations that share the same EO data directory.
    Reads EO data once, then processes each combination using the shared data.
    Returns list of (combination, success_boolean) tuples.
    """
    results = []
    country, scale, crop_folder = group_key

    # Create one GeoMerge instance to read EO data
    gm = GeoMerge(path_config_file)
    gm.parse_config("DEFAULT")

    if parallel:
        gm.logger = log.SafeLogger(
            name=f"geoprepare.merge.{country}",
            level=gm.compute_logging_level(),
        )

    # Initialize with the first combination to set up merge_eo_files() dependencies
    first_combo = group_combos[0]
    _, _, first_crop, first_season = first_combo
    gm.country_information(country, scale, first_crop, first_season)

    # Override crop_folder if using cropland mask (merge_eo_files checks use_cropland_mask)
    gm.logger.info(f"Reading EO data once for group {group_key} ({len(group_combos)} combinations)")

    # Read EO data ONCE for the whole group
    df_eo = gm.merge_eo_files()

    # Process each combination using the shared EO data
    for combo in group_combos:
        try:
            result = process_combination(combo, path_config_file, parallel, df_eo=df_eo)
            results.append(result)
        except Exception as e:
            if parallel:
                gm.logger.error(f"[ERROR] Combination {combo}: {e}")
            else:
                print(f"[ERROR] Combination {combo}: {e}")
            results.append((combo, False))

    return results


def _group_combinations(parser, all_combinations):
    """
    Group combinations by EO data key (country, scale, crop_folder).
    Combinations in the same group share the same EO data directory
    and only need to read it once.
    """
    from collections import defaultdict
    groups = defaultdict(list)

    for combo in all_combinations:
        country, scale, crop, _ = combo
        use_cropland_mask = parser.getboolean(country, "use_cropland_mask")
        crop_folder = "cr" if use_cropland_mask else crop
        key = (country, scale, crop_folder)
        groups[key].append(combo)

    return dict(groups)


def run(path_config_file=["geobase.txt", "geoextract.txt"]):
    """
    Main function that can run either sequentially or in parallel,
    depending on the `parallel` argument.

    Combinations that share the same EO data directory are grouped together
    so the EO data is read only once per group.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 1. Instantiate GeoMerge just to parse the config and create run combos
    gm_master = GeoMerge(path_config_file)
    gm_master.parse_config("DEFAULT")

    # 2. Get all combinations and group by EO data key
    all_combinations = gm_master.create_run_combinations()
    groups = _group_combinations(gm_master.parser, all_combinations)

    # Build per-country crop list from all_combinations for display
    crops_by_country = {}
    for country in gm_master.countries:
        crops_by_country[country] = sorted({c[2] for c in all_combinations if c[0] == country})
    crops_display = " | ".join(
        f"{c}: {', '.join(crops)}" for c, crops in crops_by_country.items()
    )

    from . import utils
    utils.display_run_summary("GeoMerge Runner", [
        ("Usage", "from geoprepare import geomerge; geomerge.run(cfg)"),
        ("cfg", "[geobase.txt, countries.txt, crops.txt, geoextract.txt]"),
        ("Countries", gm_master.countries),
        ("Crops", crops_display),
        ("Years", f"{gm_master.start_year} - {gm_master.end_year}"),
        ("Combinations", str(len(all_combinations))),
        ("EO groups", f"{len(groups)} (EO data read once per group)"),
        ("Parallel", str(gm_master.parallel_merge)),
        ("Intermed dir", str(gm_master.dir_intermed)),
        ("Output dir", str(gm_master.dir_output)),
    ])

    results = []

    if not gm_master.parallel_merge:
        # -- SEQUENTIAL EXECUTION --
        for key, group_combos in tqdm(groups.items(), desc="Processing groups"):
            group_results = process_group(key, group_combos, path_config_file)
            results.extend(group_results)
    else:
        # -- PARALLEL EXECUTION (one process per EO group) --
        num_cpus = int(gm_master.fraction_cpus * (os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            future_to_key = {
                executor.submit(process_group, key, group_combos, path_config_file, True): key
                for key, group_combos in groups.items()
            }

            for future in tqdm(
                as_completed(future_to_key),
                total=len(future_to_key),
                desc="Processing groups (parallel)",
                leave=True,
            ):
                key = future_to_key[future]
                try:
                    group_results = future.result()
                    results.extend(group_results)
                except Exception as e:
                    print(f"[ERROR] Group {key}: {e}")
                    results.extend([(c, False) for c in groups[key]])

    succeeded = sum(s for _, s in results)
    failed = [(c, s) for c, s in results if not s]
    print(f"\nGeoMerge: {succeeded}/{len(results)} combinations produced output.")
    if failed:
        print(f"  Failed: {[c for c, _ in failed]}")


if __name__ == "__main__":
    # Folder structure is as follows:
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<CROP>\
    # <base_dir>\input\crop_t10\<COUNTRY>\<SCALE>\<EO_DATA_FILE.csv>
    run()