# README ##############################################################################################################
# 1. Read in files for EO variables (NDVI, soil moisture, precipitation, etc.) from /cmongp1/GEOGLAM/Input/intermed
# 2. For each admin_1 produce a single csv file per year, by multiplying the variable value in each grid cell by the
# percentage of crop area in that grid cell
# 3. Apply a threshold based on either floor or ceil flags
#    3.a floor: select all grid cells in admin_1 with crop area above this precentage
#    3.b ceil: select all grid cells in admin_1 with crop area above this percentile
# 4. Output csv file to constants_base.dir_all_inputs + os.sep + always.dir_crop_inputs (/cmongp1/GEOGLAM/Input/crop_*)
#######################################################################################################################
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

np.seterr(
    invalid="ignore"
)  # HACK! Ignore 'RuntimeWarning: invalid value encountered in ...'.


def get_var_fname(params, var, year, doy):
    """
    determines the infile name
    :param var:
    :param year:
    :param doy:
    :return:
    """
    year_doy = f"{year}{str(doy).zfill(3)}"

    if var in ["ndvi", "gcvi"]:
        fname = f"mod09.{var}.global_0.05_degree.{year}.{str(doy).zfill(3)}.c6.v1.tif"
    elif var == "esi_4wk":
        fname = f"esi_dfppm_4wk_{year_doy}.tif"
    elif var == "esi_12wk":
        fname = f"esi_dfppm_12wk_{year_doy}.tif"
    elif var == "ncep2_precip":
        fname = f"ncep2_{year_doy}_precip_global.tif"
    elif var == "ncep2_mean":
        fname = (
            "ncep2_temp"
            + os.sep
            + "mean"
            + os.sep
            + "ncep2_"
            + year_doy
            + "_mean_global.tif"
        )
    elif var == "ncep2_min":
        fname = (
            "ncep2_temp"
            + os.sep
            + "min"
            + os.sep
            + "ncep2_"
            + year_doy
            + "_min_global.tif"
        )
    elif var == "ncep2_max":
        fname = (
            "ncep2_temp"
            + os.sep
            + "max"
            + os.sep
            + "ncep2_"
            + year_doy
            + "_max_global.tif"
        )
    elif var == "cpc_tmax":
        fname = f"cpc_{year_doy}_tmax_global.tif"
    elif var == "cpc_tmin":
        fname = f"cpc_{year_doy}_tmin_global.tif"
    elif var == "cpc_precip":
        fname = f"cpc_{year_doy}_precip_global.tif"
    elif var == "soil_moisture_as1":
        fname = f"nasa_usda_soil_moisture_{year_doy}_as1_global.tif"
    elif var == "soil_moisture_as2":
        fname = f"nasa_usda_soil_moisture_{year_doy}_as2_global.tif"
    elif var == "lai":
        fname = f"MCD15A2H.A{year_doy}_Lai_500m_mosaic_0p05.tif"
    elif var == "fpar":
        fname = f"MCD15A2H.A{year_doy}_Fpar_500m_mosaic_0p05.tif"
    elif var == "chirps":
        fname = Path("global") / f"chirps_v2.0.{year_doy}_global.tif"
    elif var == "chirps_gefs":
        filelist = glob.glob(
            str(params.dir_interim) + os.sep + "chirps_gefs" + os.sep + "*.tif"
        )
        fname = os.path.basename(filelist[0])
    elif var == "smos" and year_doy >= "2015124":
        fname = "SM_OPER_MIR_CLF33A_" + year_doy + ".tif"
    elif var == "smos" and year_doy < "2015124":
        fname = "SM_RE04_MIR_CLF33A_" + year_doy + ".tif"
    elif var == "lst":
        fname = f"MOD11C1.A{year_doy}_global.tif"
    else:
        params.logger.error("Variable " + var + " does not exist")

    return fname


def remove_duplicates(list_of_tuples):
    """
    Discard duplicates from 'list_of_tuples', where each item is a tuple.
    Two tuples are considered duplicates if they match in every position
    except for positions containing a GeoDataFrame. Any GeoDataFrame inside
    the tuple is ignored (replaced with a placeholder when checking duplicates).

    Returns a list of tuples where duplicates are removed.
    """
    seen = set()
    result = []

    for item in list_of_tuples:
        # item is a tuple. Build a hashable "key" by replacing any GeoDataFrame with None.
        # If there's *other* unhashable stuff, you might need to handle that too,
        # but for now we'll assume only GDF can cause problems.
        comparison_key = []
        for val in item:
            if isinstance(val, gp.GeoDataFrame):
                # Replace GDF with None (or any placeholder)
                comparison_key.append(None)
            else:
                comparison_key.append(val)

        # Convert the list to a tuple so it's hashable
        comparison_key = tuple(comparison_key)

        # Check if we've seen this "key" before
        if comparison_key not in seen:
            seen.add(comparison_key)
            result.append(item)

    return result


def process(val):
    from .stats import geom_extract

    params, country, crop, scale, var, year, afi_file, df_country = val
    if scale not in ["admin_1", "admin_2"]:
        raise ValueError(
            f"Scale {scale} is not valid, should be either admin_1 or admin_2"
        )

    # Do not process CHIRPS forecast data if it is not for the current year
    if var == "chirps_gefs" and year != ar.utcnow().year:
        return

    threshold = params.parser.getboolean(country, "threshold")
    limit = utils.crop_mask_limit(params, country, threshold)

    dir_crop_inputs = Path(f"crop_t{limit}") if threshold else Path(f"crop_p{limit}")

    dir_output = params.dir_input / dir_crop_inputs / country / scale / crop / var
    os.makedirs(dir_output, exist_ok=True)

    current_year = ar.utcnow().year
    end_doy = 367 if calendar.isleap(year) else 366

    # Loop over df_country
    for idx, row in df_country.iterrows():
        nan_str = f"{np.NaN},{np.NaN},{np.NaN},{np.NaN},{np.NaN},{np.NaN}"
        hndl_outf = None
        region = row["ADM1_NAME"]
        region_id = row["ADM1_ID"]
        stat_str = []

        # lowercase region name and replace spaces with underscores
        region = region.str.lower().str.replace(" ", "_")

        path_output = dir_output / Path(f"{region}_{region_id}_{year}_{var}_{crop}.csv")

        # Check if we are processing current year or in case of a previous year the REDO flag is False
        # If so then only modify those lines where we do not have data currently
        use_partial_file = current_year == year or (current_year > year and not params.redo)

        if use_partial_file and os.path.isfile(path_output):
            with open(path_output) as hndl_outf:
                reader = csv.reader(hndl_outf)

                # Exclude the header and store remaining rows into a list
                list_rows = list(reader)[1:]

        for doy in range(1, end_doy):
            if var == "chirps_gefs":
                if doy > 1:
                    break

                forecast_date = ar.utcnow().shift(days=+15).date()
                doy = forecast_date.timetuple().tm_yday

                # Process a single date for chirps_gefs
                empty_str = (
                    f"{country},{region},{region_id},{forecast_date.year},{doy},{nan_str}"
                )
            else:
                if use_partial_file and hndl_outf and list_rows[doy - 1][5] != "nan":
                    stat_str.append(",".join(list_rows[doy - 1]))
                    continue

                empty_str = f"{country},{region},{region_id},{year},{doy},{nan_str}"

            fl_var = (
                params.dir_interim
                / var
                / Path(get_var_fname(params, var, year, doy))
            )
            if os.path.isfile(fl_var):
                val = geom_extract(row["geometry"], var, fl_var, stats_out=['mean', 'counts'], afi=afi_file, afi_thresh=2000, thresh_type='Fixed')
                # Extract values from the nested dictionary
                if val:
                    values = list(val['stats'].values()) + list(val['counts'].values())
                    # Convert to a comma-separated string
                    comma_separated_str = ', '.join(map(str, values))
                    stat_str.append(f"{country},{region},{region_id},{year},{doy},{comma_separated_str}")
                else:
                    stat_str.append(empty_str)
            else:
                stat_str.append(empty_str)
        # Add a header and store as pandas dataframe
        stat_str.insert(
            0,
            f"country,region,region_id,year,doy,{var},num_pixels,average_crop_percentage,"
            "median_crop_percentage,min_crop_percentage,max_crop_percentage",
        )

        df = pd.read_csv(io.StringIO("\n".join(stat_str)))
        df.to_csv(path_output, index=False)

    return stat_str

def run(params):
    """
    .
     Returns:

    """
    list_combinations = []
    num_cpus = int(params.fraction_cpus * cpu_count()) if params.parallel_process else 1
    years = list(range(params.start_year, params.end_year + 1))

    for country in params.countries:
        use_cropland_mask = params.parser.getboolean(country, "use_cropland_mask")
        crops = ast.literal_eval(params.parser.get(country, "crops"))
        vars = ast.literal_eval(params.parser.get(country, "eo_model"))
        scales = ast.literal_eval(params.parser.get(country, "scales"))
        df_cmask = gp.GeoDataFrame.from_file(
            params.dir_regions_shp / params.parser.get(country, "shp_boundary"),
            engine="pyogrio",
        )

        # Rename ADMIN0 to ADM0_NAME and ADMIN1 to ADM1_NAME and ADMIN2 to ADM2_NAME
        df_cmask.rename(
            columns={
                "ADMIN0": "ADM0_NAME",
                "ADMIN1": "ADM1_NAME",
                "ADMIN2": "ADM2_NAME",
                "name1": "ADM1_NAME",
                "name0": "ADM0_NAME",
                "asap1_id": "ADM1_ID",
                "asap0_id": "ADM0_ID",
            },
            inplace=True
        )

        # Extract for country
        df_country = df_cmask[
            df_cmask["ADM0_NAME"].str.lower().str.replace(" ", "_") == country
        ]

        for crop in crops:
            if use_cropland_mask:
                path_mask = params.parser.get(country, "mask")
            else:
                path_mask = params.parser.get(crop, "mask")
            path_mask = params.dir_masks / path_mask

            for scale in scales:
                name_crop = "cr" if use_cropland_mask else crop
                for var in vars:
                    list_combinations.extend(
                        list(
                            itertools.product(
                                [params],
                                [country],
                                [name_crop],
                                [scale],
                                [var],
                                years,
                                [path_mask],
                                [df_country],
                            )
                        )
                    )

    list_combinations = remove_duplicates(list_combinations)

    params.logger.error("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    params.logger.error(params.countries)
    params.logger.error(f"Spatial scale: {scales}, REDO flag: {params.redo}")
    params.logger.error(f"Start: {params.start_year}, End: {params.end_year}")
    params.logger.error(f"EO vars to process: {vars}")
    params.logger.error(f"CPUs: {num_cpus if params.parallel_process else 1}")
    params.logger.error(f"Output directory: {params.dir_output}")
    params.logger.error("++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if params.parallel_process:
        with Pool(num_cpus) as p:
            with tqdm(total=len(list_combinations)) as pbar:
                for i, _ in tqdm(
                    enumerate(p.imap_unordered(process, list_combinations))
                ):
                    desc = f"Processing {list_combinations[i][1]} {list_combinations[i][2]} {list_combinations[i][4]}"
                    pbar.set_description(desc)
                    pbar.update()
    else:
        pbar = tqdm(list_combinations)
        for i, val in enumerate(pbar):
            desc = f"Processing {list_combinations[i][1]} {list_combinations[i][2]} {list_combinations[i][4]}"
            pbar.set_description(desc)
            pbar.update()

            process(val)


if __name__ == "__main__":
    pass
