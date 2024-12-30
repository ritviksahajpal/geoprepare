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
from tqdm import tqdm
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count

from .. import utils


# ========================
#         UTILITIES
# ========================
np.seterr(invalid="ignore")


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
        "chirps": f"global/chirps_v2.0.{year_doy}_global.tif",
        "chirps_gefs": f"data.{year}.{month:02d}{day_of_month:02d}.tif",
        "lst": f"MOD11C1.A{year_doy}_global.tif"
    }

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
    empty_values = ",".join([str(np.NaN)] * 6)
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


def extract_stats(geometry, var: str, fl_var: Path, afi_file, limit: int) -> str:
    """
    Calls geom_extract(...) and returns a comma-separated string of the extracted results.
    If extraction fails/returns no data, return "".
    """
    from .stats import geom_extract

    val_extracted = geom_extract(
        geometry,
        var,
        fl_var,
        stats_out=["mean", "counts"],
        afi=afi_file,
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

    # Example CSV header with 6 numeric columns for your var + counts
    header = (
        f"country,region,region_id,year,doy,{var},num_pixels,"
        "average_crop_percentage,median_crop_percentage,"
        "min_crop_percentage,max_crop_percentage"
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
    afi_file,
    limit: int,
    country: str,
    region: str,
    region_id: str,
    var: str,
    year: int,
    params
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
        fl_var = params.dir_interim / var / fname

        if not os.path.isfile(fl_var):
            daily_stats.append(empty_str)
            continue

        values_str = extract_stats(row["geometry"], var, fl_var, afi_file, limit)
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
    afi_file,
    limit: int,
    var: str,
    params,
    year: int,
    country: str,
    region: str,
    region_id: str,
    end_doy: int,
    existing_rows: list[list[str]],
    use_partial_file: bool
) -> None:
    """
    Process a 'regular' variable for each day of the year (1..end_doy).
    If partial data exists for a given day, skip re-processing.
    """
    for doy in range(1, end_doy):
        date_part = f"{year},{doy}"
        empty_str = get_default_empty_str(country, region, region_id, date_part)

        # Build the filename (adjust as needed to match your naming scheme).
        fname = get_var_fname(params, var, year, doy)

        # Inline logic for nsidc_surface and nsidc_rootzone
        if var == "nsidc_surface":
            fl_var = params.dir_interim / "nsidc" / "daily" / "surface" / fname
        elif var == "nsidc_rootzone":
            fl_var = params.dir_interim / "nsidc" / "daily" / "rootzone" / fname
        else:
            fl_var = params.dir_interim / var / fname

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

        # Extract stats
        values_str = extract_stats(row["geometry"], var, fl_var, afi_file, limit)
        if values_str:
            daily_stats.append(f"{country},{region},{region_id},{date_part},{values_str}")
        else:
            daily_stats.append(empty_str)


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

    # 2. Skip 'chirps_gefs' if year != current year
    if skip_chirps_gefs(var, year):
        return

    # 3. Prepare the output directory
    dir_output = prepare_output_directory(params, country, scale, crop, var)

    # 4. Identify admin fields
    admin_name, admin_id = get_admin_fields(scale)

    # 5. Some references
    current_year = ar.utcnow().year
    end_doy = get_end_doy(year)
    threshold = params.parser.getboolean(country, "threshold")
    limit = utils.crop_mask_limit(params, country, threshold)
    use_partial = should_use_partial_file(params, year, current_year)

    # 6. Iterate over each row (region) in df_country
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

        # 7. If var == "chirps_gefs", handle that with special logic
        if var == "chirps_gefs":
            process_chirps_gefs(
                daily_stats,
                row,
                afi_file,
                limit,
                country,
                region,
                region_id,
                var,
                year,
                params
            )
        else:
            # Otherwise, process day-by-day
            process_regular_var(
                daily_stats,
                row,
                afi_file,
                limit,
                var,
                params,
                year,
                country,
                region,
                region_id,
                end_doy,
                existing_rows,
                use_partial
            )

        # 8. Write the results to CSV
        write_daily_stats_to_csv(daily_stats, path_output, var)


# ========================
#        MAIN RUNNER
# ========================
def build_combinations(params):
    """
    Builds the list of combinations to process
    (country, crop, scale, var, year, mask, df_country, etc.).
    """
    list_combinations = []
    years = list(range(params.start_year, params.end_year + 1))

    for country in params.countries:
        category = params.parser.get(country, "category")
        use_cropland_mask = params.parser.getboolean(country, "use_cropland_mask")
        crops = ast.literal_eval(params.parser.get(country, "crops"))
        vars_list = ast.literal_eval(params.parser.get(category, "eo_model"))
        scales = ast.literal_eval(params.parser.get(country, "scales"))

        # Load your GeoDataFrame
        df_cmask = gp.read_file(
            params.dir_regions_shp / params.parser.get(country, "shp_boundary"),
            engine="pyogrio",
        )

        # Rename columns
        df_cmask.rename(
            columns={
                "ADMIN0": "ADM0_NAME",
                "ADMIN1": "ADM1_NAME",
                "ADMIN2": "ADM2_NAME",
                "name1": "ADM1_NAME",
                "name0": "ADM0_NAME",
                "FNID": "ADM_ID",
                "asap1_id": "ADM_ID",
                "asap0_id": "ADM0_ID",
            },
            inplace=True,
        )

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
            path_mask = params.dir_masks / path_mask

            for scale in scales:
                # Decide the crop name
                name_crop = "cr" if use_cropland_mask else crop

                for var in vars_list:
                    # Extend the list of parameter combos
                    # Each combination is passed as a tuple to process()
                    # Last element is df_country
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


def run(params):
    """

    """
    list_combinations = build_combinations(params)
    num_cpus = (
        int(params.fraction_cpus * cpu_count()) if params.parallel_extract else 1
    )

    # Logging info
    scales = ast.literal_eval(params.parser.get(params.countries[0], "scales"))
    vars_list = ast.literal_eval(params.parser.get(params.countries[0], "eo_model"))
    msg = (
        "++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        f"# CPUs: {num_cpus}\n"
        f"Parallel: {params.parallel_extract}\n"
        f"REDO flag: {params.redo}\n"
        f"Countries: {params.countries}\n"
        f"EO vars to process: {vars_list}\n"
        f"Start Year: {params.start_year}, End Year: {params.end_year}\n"
        f"Output directory: {params.dir_output}\n"
        "++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    params.logger.error(msg)

    if params.parallel_extract:
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
