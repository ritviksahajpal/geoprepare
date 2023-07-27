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
import datetime
import itertools
import arrow as ar
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
import bottleneck as bn
from rasterio.io import MemoryFile
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


def get_var(var, hndl_fl_var):
    """
    1. Reads in data from variable file handle
    2. Converts data type to float
    3. Assigns NaNs to values that are invalid e.g. -ve precip
    Determines the method of masking and scaling the input data
    :param var:
    :param hndl_fl_var:
    :return:
    """
    tmp_var = hndl_fl_var.read(1).astype(float)

    if var == "ndvi":
        tmp_var[(tmp_var > 250) | (tmp_var < 50)] = np.NaN
    elif var == "gcvi":
        tmp_var[(tmp_var > 200000) | (tmp_var < 0)] = np.NaN
        tmp_var /= 10000.0
    elif var in ["esi_4wk", "esi_12wk"]:
        tmp_var = (tmp_var + 4.0) * 10.0
        tmp_var[tmp_var < 0.0] = np.NaN
    elif var in ["soil_moisture_as1", "soil_moisture_as2"]:
        tmp_var[tmp_var < 0.0] = np.NaN
        tmp_var[tmp_var == 9999.0] = np.NaN
    elif var in ["chirps"]:
        tmp_var[tmp_var < 0.0] = np.NaN
        tmp_var /= 100.0
    elif var in ["chirps_gefs"]:
        tmp_var[tmp_var < 0.0] = np.NaN
        tmp_var /= 100.0
    elif var in ["cpc_tmax", "cpc_tmin"]:
        tmp_var[tmp_var < -273.15] = np.NaN
    elif var in ["lst"]:
        tmp_var = tmp_var * 0.02 - 273.15
        tmp_var[tmp_var < -123.15] = np.NaN
    else:
        tmp_var[tmp_var < 0.0] = np.NaN

    return tmp_var


def nanaverage(var, var_weights):
    """
    Compute weighted average taking NaNs into account
    :param var:
    :param var_weights:
    :return:
    """

    indices = ~np.isnan(var)

    return np.average(var[indices], weights=var_weights[indices])


def nancount(var):
    """
    Compute weighted average taking NaNs into account
    :param var:
    :param var_weights:
    :return:
    """

    indices = ~np.isnan(var)

    return np.size(var[indices])


def compute_single_stat(
    fl_var, name_var, mask_crop_per, empty_str, country, region, region_id, year, doy
):
    """

    Args:
        fl_var ():
        name_var ():
        mask_crop_per ():
        empty_str ():
        country ():
        region ():
        region_id ():
        year ():
        doy ():

    Returns:

    """
    with MemoryFile(open(fl_var, "rb").read()) as memfile:
        with memfile.open() as fl_var:
            arr_var = get_var(name_var, fl_var)
            arr_crop_var = arr_var * (mask_crop_per > 0.0)
            sum_crop_var = bn.nansum(arr_crop_var)

            # if there are crop pixels but the underlying data layer has no data then sum_crop_var will be 0
            if sum_crop_var == 0:
                out_str = empty_str
            else:
                if name_var in ["esi_4wk", "esi_12wk"]:
                    wavg_unscaled = nanaverage(arr_crop_var, mask_crop_per)
                    weighted_average = (wavg_unscaled / 10.0) - 4.0
                else:
                    weighted_average = nanaverage(arr_crop_var, mask_crop_per)

                crop_mask_weighted_average = mask_crop_per[
                    ~np.isnan(arr_crop_var)
                ]  # Crop Mask Weighted Average
                crop_mask_weighted_average[crop_mask_weighted_average <= 0.0] = np.NaN
                arr_crop_var[arr_crop_var <= 0.0] = np.NaN
                num_pixels = np.count_nonzero(
                    ~np.isnan(arr_crop_var)
                )  # Total number of pixels (after threshold is applied)
                median_crop_percentage = bn.nanmedian(
                    crop_mask_weighted_average
                )  # Median crop percentage of pixels (after threshold)
                average_crop_percentage = bn.nanmean(
                    crop_mask_weighted_average
                )  # Average crop percentage of the pixels (after threshold)
                min_crop_percentage = bn.nanmin(
                    crop_mask_weighted_average
                )  # Min crop percentage of pixels (after threshold)
                max_crop_percentage = bn.nanmax(
                    crop_mask_weighted_average
                )  # Max crop percentage of the pixels (after threshold)

                out_str = (
                    f"{country},{region},{region_id},{year},{doy},{weighted_average},{num_pixels},{average_crop_percentage},"
                    f"{median_crop_percentage},{min_crop_percentage},{max_crop_percentage}"
                )

    return out_str


def compute_stats(
    params, country, region, region_id, year, name_var, mask_crop_per, path_outf
):
    """

    Args:
        params ():
        country ():
        region ():
        region_id ():
        year ():
        name_var ():
        mask_crop_per ():
        path_outf ():

    Returns:

    """
    current_year = ar.utcnow().year
    end_doy = 367 if calendar.isleap(year) else 366

    stat_str = []
    nan_str = f"{np.NaN},{np.NaN},{np.NaN},{np.NaN},{np.NaN},{np.NaN}"
    hndl_outf = None

    # Check if we are processing current year or in case of a previous year the REDO flag is False
    # If so then only modify those lines where we do not have data currently
    use_partial_file = current_year == year or (current_year > year and not params.redo)

    if use_partial_file and os.path.isfile(path_outf):
        with open(path_outf) as hndl_outf:
            reader = csv.reader(hndl_outf)

            # Exclude the header and store remaining rows into a list
            list_rows = list(reader)[1:]

    stat_str = []
    for doy in range(1, end_doy):
        if name_var == "chirps_gefs":
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
            / name_var
            / Path(get_var_fname(params, name_var, year, doy))
        )
        if not os.path.isfile(fl_var):
            out_str = empty_str
        else:
            try:
                out_str = compute_single_stat(
                    fl_var,
                    name_var,
                    mask_crop_per,
                    empty_str,
                    country,
                    region,
                    region_id,
                    year,
                    doy,
                )
            except Exception as e:
                params.logger.error(f"{e} {fl_var} {name_var} {country} {region}")

        stat_str.append(out_str)

    return stat_str


def setup(params, country, crop, scale, var, crop_mask, threshold, limit):
    """

    Args:
        params ():
        country ():
        scale ():
        crop ():
        var ():
        crop_mask ():
        threshold ():
        limit ():

    Returns:

    """
    # Extracting 'kansas' from 'kansas_188017000' and finding length to extract 9 digit number
    # the \d{9} bit matches exactly 9 digits. the bit of the regex that is in (?= ... ) is a lookahead,
    # which means it is not part of the actual match, but it only matches if that follows the match.
    fname = os.path.basename(crop_mask)

    if scale == "admin_1":
        region_name = re.search(r".+(?=_\d{9}_)", fname).group(0)
    else:  # admin_2
        region_name = re.search(r".+(?=_\d{12}_)", fname).group(0)

    # Extract numeric admin identifier from crop mask file name
    region_id = re.findall(r"\d+", fname)[0]

    dir_crop_inputs = Path(f"crop_t{limit}") if threshold else Path(f"crop_p{limit}")

    dir_output = params.dir_input / dir_crop_inputs / country / scale / crop / var
    os.makedirs(dir_output, exist_ok=True)

    return region_name, region_id, dir_output


def process(val):
    """

    Args:
        val ():
        params:
        country:
        crop:
        var:
        year:
        crop_mask: name of crop mask file

    Returns:

    """
    params, country, crop, scale, var, year, crop_mask = val

    if scale not in ["admin_1", "admin_2"]:
        raise ValueError(
            f"Scale {scale} is not valid, should be either admin_1 or admin_2"
        )

    # Do not process CHIRPS forecast data if it is not for the current year
    if var == "chirps_gefs" and year != ar.utcnow().year:
        return

    threshold = params.parser.getboolean(country, "threshold")
    limit = utils.crop_mask_limit(params, country, threshold)
    region, region_id, dir_output = setup(
        params, country, crop, scale, var, crop_mask, threshold, limit
    )
    path_output = dir_output / Path(f"{region}_{region_id}_{year}_{var}_{crop}.csv")

    os.makedirs(dir_output, exist_ok=True)

    # Process variable:
    # 1. if output csv does not exist OR
    # 2. if processing current year OR
    # 3. if REDO flag is set to true
    process_current_year = datetime.datetime.now().year == year

    if not os.path.isfile(path_output) or process_current_year or params.redo:
        with MemoryFile(open(crop_mask, "rb").read()) as memfile:
            with memfile.open() as hndl_crop_mask:
                mask_crop_per = hndl_crop_mask.read(1).astype(float)

                if threshold:
                    mask_crop_per[
                        mask_crop_per < limit
                    ] = 0.0  # Create crop mask and mask pixel LT CP
                else:  # percentile
                    val_percentile = np.percentile(
                        mask_crop_per[mask_crop_per > 0.0], limit
                    )
                    mask_crop_per[mask_crop_per < val_percentile] = 0.0

                if np.count_nonzero(mask_crop_per):  # if there are no pixels then skip
                    tmp_str = compute_stats(
                        params,
                        country,
                        region,
                        region_id,
                        year,
                        var,
                        mask_crop_per,
                        path_output,
                    )

                    # Add a header and store as pandas dataframe
                    tmp_str.insert(
                        0,
                        f"country,region,region_id,year,doy,{var},num_pixels,average_crop_percentage,"
                        "median_crop_percentage,min_crop_percentage,max_crop_percentage",
                    )

                    df = pd.read_csv(io.StringIO("\n".join(tmp_str)))
                    df.to_csv(path_output, index=False)


def remove_duplicates(lst):
    return list(set([i for i in lst]))


def run(params):
    """
    .
     Returns:

    """
    all_comb = []
    num_cpus = int(params.fraction_cpus * cpu_count()) if params.parallel_process else 1
    years = list(range(params.start_year, params.end_year + 1))

    for country in params.countries:
        # Check if we use a cropland mask or not
        use_cropland_mask = params.parser.get(country, "use_cropland_mask")
        crops = ast.literal_eval(params.parser.get(country, "crops"))
        vars = ast.literal_eval(params.parser.get(country, "eo_model"))
        scales = ast.literal_eval(params.parser.get(country, "scales"))

        for crop in crops:
            for scale in scales:
                name_crop = "cr" if use_cropland_mask else crop
                path_crop_masks = params.dir_crop_masks / country / name_crop / scale
                crop_masks = list(path_crop_masks.glob(f"*_{name_crop}_crop_mask.tif"))

                if len(crop_masks):
                    for var in vars:
                        all_comb.extend(
                            list(
                                itertools.product(
                                    [params],
                                    [country],
                                    [name_crop],
                                    [scale],
                                    [var],
                                    years,
                                    crop_masks,
                                )
                            )
                        )

    all_comb = remove_duplicates(all_comb)

    params.logger.error(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    params.logger.error(params.countries)
    params.logger.error(
        f"Spatial scale (admin_1/state or admin_2/county): {scales}, REDO flag: {params.redo}"
    )
    params.logger.error(
        f"Starting year: {params.start_year}, Ending year: {params.end_year}"
    )
    params.logger.error(f"EO vars to process: {vars}")
    params.logger.error(
        f"Number of CPUs used: {num_cpus if params.parallel_process else 1}"
    )
    params.logger.error(f"Output directory: {params.dir_output}")
    params.logger.error(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )

    if params.parallel_process:
        with Pool(num_cpus) as p:
            with tqdm(total=len(all_comb)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(process, all_comb))):
                    pbar.set_description(
                        f"Processing {all_comb[i][1]} {all_comb[i][2]} {all_comb[i][4]}"
                    )
                    pbar.update()
    else:
        # Use the code below if you want to test without parallelization or if you want to debug by using pdb
        pbar = tqdm(all_comb)
        for i, val in enumerate(pbar):
            pbar.set_description(
                f"Processing {all_comb[i][1]} {all_comb[i][2]} {all_comb[i][4]}"
            )
            pbar.update()
            process(val)


if __name__ == "__main__":
    pass
