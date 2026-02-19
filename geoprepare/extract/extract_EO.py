import os
import re
import io
import ast
import csv
import glob
import calendar
import itertools
import arrow as ar
import pandas as pd
import geopandas as gp
import rasterio
from tqdm import tqdm
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager
from functools import lru_cache

from .. import utils
from ..georegion import get_boundary_col_mapping
from .. import log as _log

import logging as _logging


def _swap_logger_for_parallel(params):
    """Replace the logzero-based logger with a process-safe stdlib logger."""
    return _log.swap_for_parallel(params)


# ========================
#         UTILITIES
# ========================
np.seterr(invalid="ignore")


class RasterCache:
    """
    A simple cache for rasterio datasets to avoid repeated file opens.
    Thread-safe for read operations, but should only be used within a single process.
    """
    
    def __init__(self):
        self._cache = {}
    
    def get(self, path):
        """Get an open rasterio dataset, opening it if not cached."""
        path_str = str(path)
        if path_str not in self._cache:
            if os.path.isfile(path_str):
                self._cache[path_str] = rasterio.open(path_str)
            else:
                return None
        return self._cache[path_str]
    
    def close_all(self):
        """Close all cached datasets."""
        for ds in self._cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self._cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False


def remove_duplicates(list_of_tuples):
    """
    Discards duplicates from 'list_of_tuples', where each item is a tuple.
    Two tuples are considered duplicates if they match in every position except for any GeoDataFrame.
    """
    seen = set()
    result = []

    for item in list_of_tuples:
        comparison_key = []
        for val in item:
            if isinstance(val, gp.GeoDataFrame):
                comparison_key.append(None)
            else:
                comparison_key.append(val)
        comparison_key = tuple(comparison_key)

        if comparison_key not in seen:
            seen.add(comparison_key)
            result.append(item)

    return result


def get_var_fname(params, var, year, doy):
    """
    Determines the input file name based on the variable and date.
    """
    year_doy = f"{year}{str(doy).zfill(3)}"
    month, day_of_month = get_month_and_day_from_day_of_year(year, doy)

    # Common variable-to-filename mapping
    variable_fnames = {
        "ndvi": f"mod09.ndvi.global_0.05_degree.{year}.{str(doy).zfill(3)}.c6.v1.tif",
        "gcvi": f"mod09.gcvi.global_0.05_degree.{year}.{str(doy).zfill(3)}.c6.v1.tif",
        "esi_4wk": f"esi_dfppm_4wk_{year_doy}.tif",
        "esi_12wk": f"esi_dfppm_12wk_{year_doy}.tif",
        "ncep2_precip": f"ncep2_{year_doy}_precip_global.tif",
        "ncep2_mean": f"ncep2_temp/mean/ncep2_{year_doy}_mean_global.tif",
        "ncep2_min": f"ncep2_temp/min/ncep2_{year_doy}_min_global.tif",
        "ncep2_max": f"ncep2_temp/max/ncep2_{year_doy}_max_global.tif",
        "cpc_tmax": f"cpc_{year_doy}_tmax_global.tif",
        "cpc_tmin": f"cpc_{year_doy}_tmin_global.tif",
        "cpc_precip": f"cpc_{year_doy}_precip_global.tif",
        "soil_moisture_as1": f"nasa_usda_soil_moisture_{year_doy}_as1_global.tif",
        "soil_moisture_as2": f"nasa_usda_soil_moisture_{year_doy}_as2_global.tif",
        "nsidc_surface": f"nasa_usda_soil_moisture_{year}_{str(doy).zfill(3)}_surface_global.tif",
        "nsidc_rootzone": f"nasa_usda_soil_moisture_{year}_{str(doy).zfill(3)}_rootzone_global.tif",
        "lai": f"MCD15A2H.A{year_doy}_Lai_500m_mosaic_0p05.tif",
        "fpar": f"MCD15A2H.A{year_doy}_Fpar_500m_mosaic_0p05.tif",
        "chirps_gefs": f"data.{year}.{month:02d}{day_of_month:02d}.tif",
        "lst": f"MOD11C1.A{year_doy}_global.tif"
    }

    # CHIRPS needs version-aware path to match download output structure:
    #   dir_intermed/chirps/{version}/global/{year}/chirps_{version_str}_{year}{doy}_global.tif
    # get_var_fname returns the path relative to dir_intermed/chirps/
    if var == "chirps":
        chirps_version = params.parser.get("CHIRPS", "version", fallback="v2")
        version_str = "v2.0" if chirps_version == "v2" else "v3.0"
        return f"{chirps_version}/global/{year}/chirps_{version_str}_{year_doy}_global.tif"

    # Return filename if in dictionary
    if var in variable_fnames:
        return variable_fnames[var]
    elif var == "smos":
        # Example threshold: "2015124"
        # SM_OPER* if >= 2015124, else SM_RE04*
        return (
            f"SM_OPER_MIR_CLF33A_{year_doy}.tif"
            if year_doy >= "2015124"
            else f"SM_RE04_MIR_CLF33A_{year_doy}.tif"
        )
    else:
        # If variable doesn't match any known pattern
        params.logger.error(f"Variable '{var}' does not exist.")
        return None


def get_month_and_day_from_day_of_year(year, day_of_year):
    # Create an arrow object for the first day of the given year
    start_of_year = ar.get(f"{year}-01-01")

    # Calculate the date by adding the day of the year to the start of the year
    date = start_of_year.shift(days=day_of_year - 1)  # Adjusted for zero-based indexing

    # Extract the month and day from the date
    month = date.month
    day_of_month = date.day

    return month, day_of_month


# ========================
#     CORE PROCESSING
# ========================
# ---------------------------------------------------------------------
# 1. Utility & helper functions
# ---------------------------------------------------------------------
def validate_scale(scale: str) -> None:
    """
    Ensure scale is either 'admin_1' or 'admin_2'.
    """
    if scale not in ["admin_1", "admin_2"]:
        raise ValueError(f"Scale {scale} is not valid. Must be 'admin_1' or 'admin_2'.")


def skip_chirps_gefs(var: str, year: int) -> bool:
    """
    Determine if we should skip 'chirps_gefs' if the input year != current year.
    """
    current_year = ar.utcnow().year

    return (var == "chirps_gefs") and (year != current_year)


def get_admin_fields(scale: str) -> tuple[str, str]:
    """
    Return the admin name and ID fields based on the scale.
    """
    if scale == "admin_1":
        return "ADM1_NAME", "ADM_ID"
    else:
        return "ADM2_NAME", "ADM_ID"


def prepare_output_directory(params, country: str, scale: str, crop: str, var: str) -> Path:
    """
    Create (if needed) and return the output directory path for final CSV outputs.
    """
    threshold = params.parser.getboolean(country, "threshold")
    limit = utils.crop_mask_limit(params, country, threshold)

    dir_crop_inputs = Path(f"crop_t{limit}") if threshold else Path(f"crop_p{limit}")
    # e.g. /some/path/Output/crop_t20/angola/admin_1/cr/ndvi
    dir_output = params.dir_output / dir_crop_inputs / country / scale / crop / var
    os.makedirs(dir_output, exist_ok=True)

    return dir_output


def build_output_path(dir_output: Path, region: str, region_id: str, year: int, var: str, crop: str) -> Path:
    """
    Construct the filename for the CSV output.
    Adjust this logic to match your desired output file naming convention.
    """
    filename = f"{region_id}_{region}_{year}_{var}_{crop}.csv"

    return dir_output / filename


def build_existing_rows(path_output: Path) -> list[list[str]]:
    """
    If a CSV file already exists, read its rows (skipping the header) for partial data usage.
    """
    if not path_output.is_file():
        return []
    with path_output.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows[1:] if len(rows) > 1 else []


def should_use_partial_file(params, year: int, current_year: int) -> bool:
    """
    Decide if we should use partial file data when re-processing.
    """
    # We do so if year == current_year, or if year < current_year and we are not redoing everything.
    return (current_year == year) or (current_year > year and not params.redo)


def get_end_doy(year: int) -> int:
    """
    Return 367 if leap year, else 366.
    """
    return 367 if calendar.isleap(year) else 366


def get_default_empty_str(country: str, region: str, region_id: str, date_part: str) -> str:
    """
    Return a CSV line with NaN placeholders if no data was found.
    """
    # You mentioned 6 numeric columns: mean, counts, etc.
    empty_values = ",".join([str(np.nan)] * 6)
    return f"{country},{region},{region_id},{date_part},{empty_values}"


def read_or_skip_existing_day(doy: int, existing_rows: list[list[str]]) -> str | None:
    """
    If partial file has data for this day-of-year (index = doy-1), return it. Otherwise None.
    """
    try:
        # The 6th position (index=5) is the first numeric column in your CSV.
        if existing_rows[doy - 1][5] != "":
            return ",".join(existing_rows[doy - 1])
    except IndexError:
        pass
    return None


def extract_stats(geometry, var: str, fl_var, afi_ds, limit: int) -> str:
    """
    Calls geom_extract(...) and returns a comma-separated string of the extracted results.
    If extraction fails/returns no data, return "".
    
    Args:
        geometry: Shapely geometry for the region
        var: Variable name (e.g., 'ndvi', 'chirps')
        fl_var: Path to the indicator raster file
        afi_ds: Open rasterio dataset for the AFI/crop mask (or path if not cached)
        limit: Threshold value for crop mask filtering
    """
    from .stats import geom_extract

    val_extracted = geom_extract(
        geometry,
        var,
        fl_var,
        stats_out=["mean", "counts"],
        afi=afi_ds,
        afi_thresh=limit * 100,
        thresh_type="Fixed"
    )

    if not val_extracted:
        return ""

    # Combine "stats" and "counts"
    values = list(val_extracted["stats"].values()) + list(val_extracted["counts"].values())

    return ",".join(map(str, values))


def write_daily_stats_to_csv(daily_stats: list[str], path_output: Path, var: str) -> None:
    """
    If daily_stats is non-empty, insert a header and write to CSV.
    """
    if not daily_stats:
        return

    # CSV header matching geom_extract output: stats(mean) + counts(total, valid_data,
    # valid_data_after_masking, weight_sum, weight_sum_used)
    header = (
        f"country,region,region_id,year,doy,{var},"
        "total_pixels,valid_data,valid_data_after_masking,"
        "weight_sum,weight_sum_used"
    )
    daily_stats.insert(0, header)

    df_result = pd.read_csv(io.StringIO("\n".join(daily_stats)))
    df_result.to_csv(path_output, index=False)


# ---------------------------------------------------------------------
# 2. Special-case function for chirps_gefs
# ---------------------------------------------------------------------
def process_chirps_gefs(
    daily_stats: list[str],
    row,
    afi_ds,
    limit: int,
    country: str,
    region: str,
    region_id: str,
    var: str,
    year: int,
    params,
    raster_cache: RasterCache
) -> None:
    """
    Handle the special chirps_gefs logic: from start_date to end_date (a ~15-day window),
    either extract stats or fill with NaN if the file doesn't exist.
    """
    start_date = ar.utcnow().date().timetuple().tm_yday
    end_date = ar.utcnow().shift(days=+15).date().timetuple().tm_yday

    for jd in range(start_date, end_date):
        date_part = f"{ar.utcnow().year},{jd}"
        empty_str = get_default_empty_str(country, region, region_id, date_part)

        # Suppose you have a function that knows how to build the filename:
        # get_var_fname(params, var, year, jd) -> str
        fname = get_var_fname(params, var, year, jd)
        fl_var = params.dir_intermed / var / fname

        if not os.path.isfile(fl_var):
            daily_stats.append(empty_str)
            continue

        # Open indicator raster via cache
        indicator_ds = raster_cache.get(fl_var)
        if indicator_ds is None:
            daily_stats.append(empty_str)
            continue

        values_str = extract_stats(row["geometry"], var, indicator_ds, afi_ds, limit)
        if values_str:
            daily_stats.append(f"{country},{region},{region_id},{date_part},{values_str}")
        else:
            daily_stats.append(empty_str)


# ---------------------------------------------------------------------
# 3. General daily processor for variables other than chirps_gefs
# ---------------------------------------------------------------------
def process_regular_var(
    daily_stats: list[str],
    row,
    afi_ds,
    limit: int,
    var: str,
    params,
    year: int,
    country: str,
    region: str,
    region_id: str,
    end_doy: int,
    existing_rows: list[list[str]],
    use_partial_file: bool,
    raster_cache: RasterCache
) -> None:
    """
    Process a 'regular' variable for each day of the year (1..end_doy).
    If partial data exists for a given day, skip re-processing.
    Uses raster_cache to avoid repeated file opens for indicator rasters.
    """
    for doy in range(1, end_doy):
        date_part = f"{year},{doy}"
        empty_str = get_default_empty_str(country, region, region_id, date_part)

        # Build the filename (adjust as needed to match your naming scheme).
        fname = get_var_fname(params, var, year, doy)

        # Inline logic for nsidc_surface and nsidc_rootzone
        if var == "nsidc_surface":
            fl_var = params.dir_intermed / "nsidc" / "daily" / "surface" / fname
        elif var == "nsidc_rootzone":
            fl_var = params.dir_intermed / "nsidc" / "daily" / "rootzone" / fname
        else:
            fl_var = params.dir_intermed / var / fname

        # If the file doesn't exist, store an empty row
        if not os.path.isfile(fl_var):
            daily_stats.append(empty_str)
            continue

        # If partial data exists for this day, skip re-calculation
        if use_partial_file and existing_rows:
            existing_line = read_or_skip_existing_day(doy, existing_rows)
            if existing_line:
                daily_stats.append(existing_line)
                continue

        # Open indicator raster via cache to avoid repeated opens across regions
        indicator_ds = raster_cache.get(fl_var)
        if indicator_ds is None:
            daily_stats.append(empty_str)
            continue

        # Extract stats - pass open datasets for both indicator and AFI
        values_str = extract_stats(row["geometry"], var, indicator_ds, afi_ds, limit)
        if values_str:
            daily_stats.append(f"{country},{region},{region_id},{date_part},{values_str}")
        else:
            daily_stats.append(empty_str)


# ---------------------------------------------------------------------
# 3b. AEF processor (annual 64-band embeddings)
# ---------------------------------------------------------------------
AEF_NUM_BANDS = 64


def process_aef(params, country, crop, scale, afi_file, df_country):
    """
    Extract crop-masked zonal means from the average AEF 64-band TIF.

    Uses aef_avg_global.tif (multi-year average) instead of per-year files.
    For each admin region, computes the mean of each band over crop-masked pixels.

    Output CSV columns: country, region, region_id, aef_1 .. aef_64
    """
    from rasterio.mask import mask as rio_mask

    # 1. Validate scale and prepare output directory
    validate_scale(scale)
    dir_output = prepare_output_directory(params, country, scale, crop, "aef")
    admin_name, admin_id = get_admin_fields(scale)

    # 2. Build AEF average file path
    country_slug = country.lower().replace(" ", "_")
    aef_path = params.dir_intermed / "aef" / country_slug / "aef_avg_global.tif"
    if not aef_path.exists():
        params.logger.warning(f"AEF average file not found: {aef_path}")
        return

    # 3. Crop mask threshold
    threshold = params.parser.getboolean(country, "threshold")
    limit = utils.crop_mask_limit(params, country, threshold)
    afi_thresh = limit * 100

    # 4. Open both rasters once, iterate over regions
    aef_cols = [f"aef_{i}" for i in range(1, AEF_NUM_BANDS + 1)]
    header = ",".join(["country", "region", "region_id"] + aef_cols)

    with rasterio.open(aef_path) as aef_src, rasterio.open(afi_file) as afi_src:
        for _, row in df_country.iterrows():
            if not row[admin_name]:
                continue

            region = row[admin_name].lower().replace(" ", "_")
            region_id = row[admin_id]

            path_output = dir_output / f"{region}_{region_id}_aef_{crop}.csv"
            if path_output.exists() and not params.redo:
                continue

            geom = [row.geometry.__geo_interface__]

            try:
                # Read all 64 bands clipped to region: shape (64, h, w)
                aef_data, _ = rio_mask(aef_src, geom, crop=True, all_touched=True, nodata=np.nan)
                # Read crop mask clipped to same region: shape (1, h, w)
                afi_data, _ = rio_mask(afi_src, geom, crop=True, all_touched=True, nodata=0)
            except Exception as e:
                params.logger.warning(f"AEF mask failed for {region}: {e}")
                continue

            # Apply crop mask threshold
            crop_valid = afi_data[0] >= afi_thresh

            # Compute mean per band over crop-valid pixels
            band_means = []
            for b in range(AEF_NUM_BANDS):
                band = aef_data[b]
                valid = ~np.isnan(band) & crop_valid
                if valid.any():
                    band_means.append(float(np.nanmean(band[valid])))
                else:
                    band_means.append(float("nan"))

            # Write single-row CSV for this region
            values = ",".join(map(str, band_means))
            line = f"{country},{region},{region_id},{values}"
            df_result = pd.read_csv(io.StringIO(f"{header}\n{line}"))
            df_result.to_csv(path_output, index=False)


# ---------------------------------------------------------------------
# 4. Main entry point: process(...)
# ---------------------------------------------------------------------
def process(val):
    """
    Processes a single combination of:
      (params, country, crop, scale, var, year, afi_file, df_country).

    - Validates scale
    - Optionally skips chirps_gefs if not current year
    - Prepares output folder
    - Loops over each region in df_country, extracting stats
    - Writes out a CSV

    Uses RasterCache to keep the AFI (crop mask) dataset open for all regions,
    significantly reducing I/O overhead.
    """
    (
        params,
        country,
        crop,
        scale,
        var,
        year,
        afi_file,
        df_country,
    ) = val

    # 1. Validate scale
    validate_scale(scale)

    # 2. Handle AEF separately (64-band average, not daily single-band)
    if var == "aef":
        process_aef(params, country, crop, scale, afi_file, df_country)
        return

    # 3. Skip 'chirps_gefs' if year != current year
    if skip_chirps_gefs(var, year):
        return

    # 4. Prepare the output directory
    dir_output = prepare_output_directory(params, country, scale, crop, var)

    # 5. Identify admin fields
    admin_name, admin_id = get_admin_fields(scale)

    # 6. Some references
    current_year = ar.utcnow().year
    end_doy = get_end_doy(year)
    threshold = params.parser.getboolean(country, "threshold")
    limit = utils.crop_mask_limit(params, country, threshold)
    use_partial = should_use_partial_file(params, year, current_year)

    # 7. Use RasterCache for efficient dataset management
    with RasterCache() as raster_cache:
        # Open the AFI (crop mask) once for all regions
        afi_ds = raster_cache.get(afi_file)
        
        if afi_ds is None:
            params.logger.warning(f"AFI file not found: {afi_file}")
            return

        # 8. Iterate over each row (region) in df_country
        for _, row in df_country.iterrows():

            # Skip rows missing an admin name
            if not row[admin_name]:
                continue

            region = row[admin_name].lower().replace(" ", "_")
            region_id = row[admin_id]

            # Build the path for this region-year CSV
            path_output = build_output_path(dir_output, region, region_id, year, var, crop)

            # If we plan to use partial data, load existing rows
            existing_rows = build_existing_rows(path_output) if use_partial else []

            # We'll collect daily stats in this list
            daily_stats = []

            # 9. If var == "chirps_gefs", handle that with special logic
            if var == "chirps_gefs":
                process_chirps_gefs(
                    daily_stats,
                    row,
                    afi_ds,  # Pass open dataset instead of path
                    limit,
                    country,
                    region,
                    region_id,
                    var,
                    year,
                    params,
                    raster_cache
                )
            else:
                # Otherwise, process day-by-day
                process_regular_var(
                    daily_stats,
                    row,
                    afi_ds,  # Pass open dataset instead of path
                    limit,
                    var,
                    params,
                    year,
                    country,
                    region,
                    region_id,
                    end_doy,
                    existing_rows,
                    use_partial,
                    raster_cache
                )

            # 10. Write the results to CSV
            write_daily_stats_to_csv(daily_stats, path_output, var)


# ========================
#        MAIN RUNNER
# ========================
def build_combinations(params, skip_vars=None):
    """
    Builds the list of combinations to process
    (country, crop, scale, var, year, mask, df_country, etc.).
    skip_vars: set of (country, var) tuples to exclude (from validate_datasets).
    """
    if skip_vars is None:
        skip_vars = set()
    list_combinations = []
    years = list(range(params.start_year, params.end_year + 1))

    for country in params.countries:
        category = params.parser.get(country, "category")
        use_cropland_mask = params.parser.getboolean(country, "use_cropland_mask")
        crops = ast.literal_eval(params.parser.get(country, "crops"))
        vars_list = [v for v in ast.literal_eval(params.parser.get(category, "eo_model"))
                     if (country, v) not in skip_vars]
        scales = ast.literal_eval(params.parser.get(country, "scales"))

        # Load your GeoDataFrame
        df_cmask = gp.read_file(
            params.dir_boundary_files / params.parser.get(country, "shp_boundary"),
            engine="pyogrio",
        )

        # Rename columns using config-driven mapping
        shp_boundary = params.parser.get(country, "shp_boundary")
        col_rename = get_boundary_col_mapping(params.parser, shp_boundary)
        df_cmask.rename(columns=col_rename, inplace=True)

        # Validate that required columns exist after rename
        required_cols = ["ADM0_NAME", "ADM1_NAME", "ADM_ID"]
        for scale in scales:
            if scale == "admin_2":
                required_cols.append("ADM2_NAME")
        missing_cols = [c for c in required_cols if c not in df_cmask.columns]
        if missing_cols:
            params.logger.error(
                f"Shapefile for {country} is missing columns after rename: "
                f"{missing_cols}. Available columns: {list(df_cmask.columns)}"
            )
            continue

        # Filter GeoDataFrame for the specific country
        mask_country = country.lower().replace(" ", "_")
        df_country = df_cmask[
            df_cmask["ADM0_NAME"].str.lower().str.replace(" ", "_") == mask_country
        ]

        # Build mask path
        for crop in crops:
            if use_cropland_mask:
                path_mask = params.parser.get(country, "mask")
            else:
                path_mask = params.parser.get(crop, "mask")
            path_mask = params.dir_crop_masks / path_mask

            for scale in scales:
                # Decide the crop name
                name_crop = "cr" if use_cropland_mask else crop

                for var in vars_list:
                    if var == "aef":
                        # AEF uses multi-year average file; one combo, year=0 (unused)
                        combo = (
                            params,
                            country,
                            name_crop,
                            scale,
                            var,
                            0,
                            path_mask,
                            df_country,
                        )
                        list_combinations.append(combo)
                    else:
                        for year in years:
                            combo = (
                                params,
                                country,
                                name_crop,
                                scale,
                                var,
                                year,
                                path_mask,
                                df_country,
                            )
                            list_combinations.append(combo)

    # Remove duplicates
    list_combinations = remove_duplicates(list_combinations)
    return list_combinations


def _get_var_directory(params, var, country):
    """Return the intermediate directory for a given EO variable."""
    if var == "aef":
        country_slug = country.lower().replace(" ", "_")
        return params.dir_intermed / "aef" / country_slug
    elif var == "nsidc_surface":
        return params.dir_intermed / "nsidc" / "daily" / "surface"
    elif var == "nsidc_rootzone":
        return params.dir_intermed / "nsidc" / "daily" / "rootzone"
    else:
        return params.dir_intermed / var


def validate_datasets(params):
    """
    Pre-flight check: verify that intermediate data directories exist
    and contain .tif files for each EO variable and country.
    Logs warnings for missing datasets and returns the set of
    (country, var) pairs to skip during extraction.
    """
    missing = []

    for country in params.countries:
        category = params.parser.get(country, "category")
        vars_list = ast.literal_eval(params.parser.get(category, "eo_model"))

        for var in vars_list:
            # chirps_gefs is only for current year forecasts — skip validation
            if var == "chirps_gefs":
                continue

            dir_var = _get_var_directory(params, var, country)

            if var == "aef":
                # AEF extraction needs the average file specifically
                aef_avg = dir_var / "aef_avg_global.tif"
                if not aef_avg.exists():
                    missing.append((country, var, str(aef_avg)))
            elif not dir_var.exists() or not any(dir_var.rglob("*.tif")):
                missing.append((country, var, str(dir_var)))

    if missing:
        lines = [f"  {country}: {var} ({path})" for country, var, path in missing]
        msg = "Skipping missing intermediate datasets:\n" + "\n".join(lines)
        params.logger.warning(msg)

    return {(country, var) for country, var, _ in missing}


def run(params):
    """
    Main entry point for the EO extraction pipeline.
    """
    skip_vars = validate_datasets(params)
    list_combinations = build_combinations(params, skip_vars)
    num_cpus = (
        int(params.fraction_cpus * cpu_count()) if params.parallel_extract else 1
    )

    if params.parallel_extract:
        # Swap to pickle-safe logger for multiprocessing
        original_logger = _swap_logger_for_parallel(params)
        with Pool(num_cpus) as p:
            with tqdm(total=len(list_combinations)) as pbar:
                for i, _ in enumerate(p.imap_unordered(process, list_combinations)):
                    desc = (
                        f"Processing {list_combinations[i][1]} "
                        f"{list_combinations[i][2]} {list_combinations[i][4]} "
                        f"{list_combinations[i][5]}"
                    )
                    pbar.set_description(desc)
                    pbar.update()
        # Restore original logger
        params.logger = original_logger
    else:
        pbar = tqdm(list_combinations)
        for i, val in enumerate(pbar):
            desc = (
                f"Processing {list_combinations[i][1]} "
                f"{list_combinations[i][2]} {list_combinations[i][4]} "
                f"{list_combinations[i][5]}"
            )
            pbar.set_description(desc)
            pbar.update()
            process(val)


if __name__ == "__main__":
    pass