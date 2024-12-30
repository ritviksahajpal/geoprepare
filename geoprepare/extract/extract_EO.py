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


def build_output_path(dir_output, region, region_id, year, var, crop):
    """
    Generates the output path for a CSV file.
    """
    return dir_output / f"{region_id}_{region}_{year}_{var}_{crop}.csv"


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
def process(val):
    """
    Processes a single combination of (params, country, crop, scale, var, year, afi_file, df_country).
    Extracts stats from raster files, handles partial data, etc.
    """
    # Here we assume there's a local import in your real environment:
    from .stats import geom_extract  # type: ignore

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

    # Validate scale
    if scale not in ["admin_1", "admin_2"]:
        raise ValueError(f"Scale {scale} is not valid. Must be 'admin_1' or 'admin_2'.")

    admin_name = "ADM1_NAME" if scale == "admin_1" else "ADM2_NAME"
    admin_id = "ADM_ID"

    # Skip chirps_gefs if year != current year
    if var == "chirps_gefs" and year != ar.utcnow().year:
        return

    # ---------------------------
    # 1. Prepare output directory
    # ---------------------------
    threshold = params.parser.getboolean(country, "threshold")
    limit = utils.crop_mask_limit(params, country, threshold)

    dir_crop_inputs = Path(f"crop_t{limit}") if threshold else Path(f"crop_p{limit}")
    # e.g. D:/Users/ritvik/projects/GEOGLAM/Output/FEWSNET/crop_t20/angola/admin_1/cr/ndvi
    dir_output = params.dir_output / dir_crop_inputs / country / scale / crop / var
    os.makedirs(dir_output, exist_ok=True)

    # ---------------------------
    # 2. Loop over each region
    # ---------------------------
    current_year = ar.utcnow().year
    end_doy = 367 if calendar.isleap(year) else 366

    for _, row in df_country.iterrows():
        if not row[admin_name]:
            continue

        region = row[admin_name].lower().replace(" ", "_")
        region_id = row[admin_id]

        path_output = build_output_path(dir_output, region, region_id, year, var, crop)

        # Determine if partial file usage applies
        use_partial_file = (
            (current_year == year) or (current_year > year and not params.redo)
        )

        # If partial data exists, read it
        existing_rows = []
        if use_partial_file and path_output.is_file():
            with path_output.open() as hndl_outf:
                reader = csv.reader(hndl_outf)
                existing_rows = list(reader)[1:]  # skip header

        if var == "chirps_gefs":
            start_date = ar.utcnow().date().timetuple().tm_yday
            end_date = ar.utcnow().shift(days=+15).date().timetuple().tm_yday

            for jd in range(start_date, end_date):
                # Build default empty string for missing data
                empty_values = ",".join([str(np.NaN)] * 6)
                empty_str = f"{country},{region},{region_id},{date_part},{empty_values}"

                fl_var = params.dir_interim / var / Path(get_var_fname(var, year, jd))
                if not os.path.isfile(fl_var):
                    daily_stats.append(empty_str)
                else:
                    val_extracted = geom_extract(
                        row["geometry"], var, fl_var,
                        stats_out=["mean", "counts"],
                        afi=afi_file,
                        afi_thresh=limit * 100,
                        thresh_type="Fixed"
                    )
                    if val_extracted:
                        # Combine the stats and counts
                        values = list(val_extracted["stats"].values()) + list(val_extracted["counts"].values())
                        values_str = ",".join(map(str, values))
                        daily_stats.append(f"{country},{region},{region_id},{date_part},{values_str}")
                    else:
                        daily_stats.append(empty_str)
        else:
            daily_stats = []
            for doy in range(1, end_doy):
                # Handle special chirps_gefs case
                if var == "chirps_gefs":
                    # Only do day=1
                    if doy > 1:
                        break
                    forecast_date = ar.utcnow().shift(days=+15).date()
                    doy = forecast_date.timetuple().tm_yday
                    date_part = f"{forecast_date.year},{doy}"
                else:
                    date_part = f"{year},{doy}"

                # Build default empty string for missing data
                empty_values = ",".join([str(np.NaN)] * 6)
                empty_str = f"{country},{region},{region_id},{date_part},{empty_values}"

                # Attempt to find raster file
                fname = get_var_fname(params, var, year, doy)

                if var == "nsidc_surface":
                    fl_var = params.dir_interim / "nsidc" / "daily" / "surface" / fname
                elif var == "nsidc_rootzone":
                    fl_var = params.dir_interim / "nsidc" / "daily" / "rootzone" / fname
                else:
                    fl_var = params.dir_interim / var / fname

                if not os.path.isfile(fl_var):
                    daily_stats.append(empty_str)
                    continue

                # If partial file has data for this day (index = doy-1), skip
                if use_partial_file and existing_rows:
                    # 6th position (index=5) is the var data (or "")
                    try:
                        if existing_rows[doy - 1][5] != "":
                            daily_stats.append(",".join(existing_rows[doy - 1]))
                            continue
                    except:
                        breakpoint()

                val_extracted = geom_extract(
                    row["geometry"], var, fl_var,
                    stats_out=["mean", "counts"],
                    afi=afi_file,
                    afi_thresh=limit * 100,
                    thresh_type="Fixed"
                )
                if val_extracted:
                    # Combine the stats and counts
                    values = list(val_extracted["stats"].values()) + list(val_extracted["counts"].values())
                    values_str = ",".join(map(str, values))
                    daily_stats.append(f"{country},{region},{region_id},{date_part},{values_str}")
                else:
                    daily_stats.append(empty_str)

        if len(daily_stats):
            # Insert a header at the start
            header = (
                f"country,region,region_id,year,doy,{var},num_pixels,"
                "average_crop_percentage,median_crop_percentage,"
                "min_crop_percentage,max_crop_percentage"
            )
            daily_stats.insert(0, header)

            # Write results to CSV
            df_result = pd.read_csv(io.StringIO("\n".join(daily_stats)))
            df_result.to_csv(path_output, index=False)


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
