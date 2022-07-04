# README ##############################################################################################################
# 1. Read in files for EO variables (NDVI, soil moisture, precipitation, etc.) from /cmongp1/GEOGLAM/Input/intermed
# 2. For each admin1 produce a single csv file per year, by multiplying the variable value in each grid cell by the
# percentage of crop area in that grid cell
# 3. Apply a threshold based on either lower_threshold or upper_percentile flags
#    3.a lower_threshold: select all grid cells in admin1 with crop area above this precentage
#    3.b upper_percentile: select all grid cells in admin1 with crop area above this percentile
# 4. Output csv file to constants_base.dir_all_inputs + os.sep + always.dir_crop_inputs (/cmongp1/GEOGLAM/Input/crop_*)
#######################################################################################################################
import os
import pdb

import calendar
import csv
import arrow as ar
from tqdm import tqdm
from pathlib import Path

import datetime
SEP = os.sep


from rasterio.io import MemoryFile
import itertools
import glob
import numpy as np
import bottleneck as bn
from multiprocessing import Pool, cpu_count

np.seterr(invalid='ignore')  # HACK! Ignore 'RuntimeWarning: invalid value encountered in ...'.



def get_var_fname(var, year, doy):
    """
    determines the infile name
    :param var:
    :param year:
    :param doy:
    :return:
    """
    year_doy = f'{year}{str(doy).zfill(3)}'

    if var in ['ndvi', 'gcvi']:
        fname = f'mod09.{var}.global_0.05_degree.year.{str(doy).zfill(3)}.c6.v1.tif'
    elif var == 'esi_4wk':
        fname = f'esi_dfppm_4wk_{year_doy}.tif'
    elif var == 'esi_12wk':
        fname = f'esi_dfppm_12wk_{year_doy}.tif'
    elif var == 'ncep2_precip':
        fname = f'ncep2_{year_doy}_precip_global.tif'
    elif var == 'ncep2_mean':
        fname = 'ncep2_temp' + SEP + 'mean' + SEP + 'ncep2_' + year_doy + '_mean_global.tif'
    elif var == 'ncep2_min':
        fname = 'ncep2_temp' + SEP + 'min' + SEP + 'ncep2_' + year_doy + '_min_global.tif'
    elif var == 'ncep2_max':
        fname = 'ncep2_temp' + SEP + 'max' + SEP + 'ncep2_' + year_doy + '_max_global.tif'
    elif var == 'cpc_tmax':
        fname = f'cpc_{year_doy}_tmax_global.tif'
    elif var == 'cpc_tmin':
        fname = f'cpc_{year_doy}_tmin_global.tif'
    elif var == 'cpc_precip':
        fname = f'cpc_{year_doy}_precip_global.tif'
    elif var == 'soil_moisture_as1':
        fname = f'nasa_usda_soil_moisture_{year_doy}_as1_global.tif'
    elif var == 'soil_moisture_as2':
        fname = f'nasa_usda_soil_moisture_{year_doy}_as2_global.tif'
    elif var == 'lai':
        fname = f'MCD15A2H.A{year_doy}_Lai_500m_mosaic_0p05.tif'
    elif var == 'fpar':
        fname = f'MCD15A2H.A{year_doy}_Fpar_500m_mosaic_0p05.tif'
    elif var == 'chirps':
        fname = Path('global') / f'chirps_v2.0.{year_doy}_global.tif'
    elif var == 'chirps_gefs':
        filelist = glob.glob(str(constants.dir_intermed) + os.sep + 'chirps_gefs' + os.sep + '*.tif')
        fname = os.path.basename(filelist[0])
    elif var == 'smos' and year_doy >= '2015124':
        fname = 'SM_OPER_MIR_CLF33A_' + year_doy + '.tif'
    elif var == 'smos' and year_doy < '2015124':
        fname = 'SM_RE04_MIR_CLF33A_' + year_doy + '.tif'
    elif var == 'lst':
        fname = f'MOD11C1.A{year_doy}_global.tif'
    else:
        logger.error('Variable ' + var + ' does not exist')

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

    if var == 'ndvi':
        tmp_var[(tmp_var > 250) | (tmp_var < 50)] = np.NaN
    elif var == 'gcvi':
        tmp_var[(tmp_var > 200000) | (tmp_var < 0)] = np.NaN
        tmp_var /= 10000.
    elif var in ['esi_4wk', 'esi_12wk']:
        tmp_var = (tmp_var + 4.0) * 10.0
        tmp_var[tmp_var < 0.0] = np.NaN
    elif var in ['soil_moisture_as1','soil_moisture_as2']:
        tmp_var[tmp_var < 0.0] = np.NaN
        tmp_var[tmp_var == 9999.0] = np.NaN
    elif var in ['chirps']:
        tmp_var[tmp_var < 0.0] = np.NaN
        tmp_var /= 100.
    elif var in ['chirps_gefs']:
        tmp_var[tmp_var < 0.0] = np.NaN
        tmp_var /= 100.
    elif var in ['cpc_tmax', 'cpc_tmin']:
        tmp_var[tmp_var < -273.15] = np.NaN
    elif var in ['lst']:
        tmp_var=tmp_var*0.02-273.15
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


def compute_stats(adm0, adm1_name, adm1_num, year, name_var, mask_crop_per, path_outf):
    """

    :param adm0:
    :param adm1_name:
    :param adm1_num:
    :param year:
    :param name_var:
    :param mask_crop_per:
    :return:
    """
    end_jd = 367 if calendar.isleap(year) else 366  # Checking if year is leap year
    nan_str = str(np.NaN) + ',' + str(np.NaN) + ',' + str(np.NaN) + ',' + str(np.NaN) + ',' + str(np.NaN) + ',' + str(np.NaN)
    arr_str = []
    hndl_outf = None
    current_year = ar.utcnow().year

    # Check if current year or previous year and REDO flag is False AND variable != CHIRPS
    # If so then only modify those lines where we do not have data currently
    redo_part_file = (current_year == year or (current_year > year and not constants.do_redo)) and ((name_var != 'chirps') or (name_var != 'chirps_gefs'))

    if redo_part_file and os.path.isfile(path_outf):
        with open(path_outf) as hndl_outf:
            reader = csv.reader(hndl_outf)
            list_rows = list(reader)

    if name_var == 'chirps_gefs':
        forecast_date = ar.utcnow().shift(days=+15).date()

        # Process a single date for chirps_gefs
        empty_str = str(adm0) + ',' + str(adm1_name) + ',' + str(adm1_num) + ',' + str(forecast_date.year) + ',' + str(forecast_date.timetuple().tm_yday) + ',' + nan_str

        fl_var = path_vars / name_var / Path(get_var_fname(name_var, year, 1))
        pdb.set_trace()
        if not os.path.isfile(fl_var):
            out_str = empty_str
        else:
            with MemoryFile(open(fl_var, 'rb').read()) as memfile:
                with memfile.open() as fl_var:
                    arr_var = get_var(name_var, fl_var)
                    arr_crop_var = arr_var * (mask_crop_per > 0.0)
                    sum_crop_var = bn.nansum(arr_crop_var)

                    # if there are crop pixels but the underlying data layer has no data then sum_crop_var will be 0
                    if sum_crop_var == 0:
                        out_str = empty_str
                    else:
                        wavg = nanaverage(arr_crop_var, mask_crop_per)
                        # print(jd)
                        cpwe = mask_crop_per[~np.isnan(arr_crop_var)]  # Crop Mask Weighted Average
                        cpwe[cpwe <= 0.0] = np.NaN
                        arr_crop_var[arr_crop_var <= 0.0] = np.NaN
                        num = np.count_nonzero(~np.isnan(arr_crop_var))  # Total number of pixels (after threshold)
                        med = bn.nanmedian(cpwe)  # Median crop percentage of pixels (after threshold)
                        av = bn.nanmean(cpwe)  # Average crop percentage of the pixels (after threshold)
                        min_pix = bn.nanmin(cpwe)  # Min crop percentage of pixels (after threshold)
                        max_pix = bn.nanmax(cpwe)  # Max crop percentage of the pixels (after threshold)

                        # var, var + '_tot_pix', var + '_wa_crop', var + '_wmed_crop', var + '_min_crop', var + '_max_crop'
                        out_str = str(adm0) + ',' + \
                                  str(adm1_name) + ',' + \
                                  str(adm1_num) + ',' + \
                                  str(forecast_date.year) + ',' + \
                                  str(forecast_date.timetuple().tm_yday) + ',' + \
                                  str(wavg) + ',' + \
                                  str(num) + ',' + \
                                  str(av) + ',' + \
                                  str(med) + ',' + \
                                  str(min_pix) + ',' + \
                                  str(max_pix)

        arr_str.append(out_str)
        return arr_str
    else:
        arr_str = []
        for jd in range(1, end_jd):
            if redo_part_file and hndl_outf and list_rows[jd - 1][5] != 'nan':
                arr_str.append(','.join(list_rows[jd - 1]))
                continue

            empty_str = str(adm0) + ',' + str(adm1_name) + ',' + str(adm1_num) + ',' + str(year) + ',' + str(jd) + ',' + nan_str

            fl_var = path_vars / Path(get_var_fname(name_var, year, jd))
            if not os.path.isfile(fl_var):
                out_str = empty_str
            else:
                with MemoryFile(open(fl_var, 'rb').read()) as memfile:
                    with memfile.open() as fl_var:
                        arr_var = get_var(name_var, fl_var)
                        arr_crop_var = arr_var * (mask_crop_per > 0.0)
                        sum_crop_var = bn.nansum(arr_crop_var)

                        # if there are crop pixels but the underlying data layer has no data then sum_crop_var will be 0
                        if sum_crop_var == 0:
                            out_str = empty_str
                        else:
                            if name_var in ['esi_4wk', 'esi_12wk']:
                                wavg_unscaled = nanaverage(arr_crop_var, mask_crop_per)
                                wavg = (wavg_unscaled / 10.0) - 4.0
                            else:
                                wavg = nanaverage(arr_crop_var, mask_crop_per)
                            # print(jd)
                            cpwe = mask_crop_per[~np.isnan(arr_crop_var)]  # Crop Mask Weighted Average
                            cpwe[cpwe <= 0.0] = np.NaN
                            arr_crop_var[arr_crop_var <= 0.0] = np.NaN
                            num = np.count_nonzero(~np.isnan(arr_crop_var))  # Total number of pixels (after threshold)
                            med = bn.nanmedian(cpwe)  # Median crop percentage of pixels (after threshold)
                            av = bn.nanmean(cpwe)  # Average crop percentage of the pixels (after threshold)
                            min_pix = bn.nanmin(cpwe)  # Min crop percentage of pixels (after threshold)
                            max_pix = bn.nanmax(cpwe)  # Max crop percentage of the pixels (after threshold)

                            out_str = str(adm0) + ',' + \
                                      str(adm1_name) + ',' + \
                                      str(adm1_num) + ',' + \
                                      str(year) + ',' + \
                                      str(jd) + ',' + \
                                      str(wavg) + ',' + \
                                      str(num) + ',' + \
                                      str(av) + ',' + \
                                      str(med) + ',' + \
                                      str(min_pix) + ',' + \
                                      str(max_pix)

            arr_str.append(out_str)

        return arr_str


def process(val):
    """
    val: adm0, crop, var
    adm0 can refer to country i.e. admin 0 level region
    :param val:
    :return:
    """
    adm0, crop, var, yr, crop_mask = val[0], val[1], val[2], val[3], val[4]

    if var == 'chirps_gefs' and yr != ar.utcnow().year:
        return
    # Get admin level 1 region e.g. 'kansas_188017000' from 'kansas_188017000_ww_crop_mask.tif'
    adm1 = os.path.basename(crop_mask)[:-17]
    crop_name = os.path.basename(crop_mask)[-16:-14]
    # Extracting 'kansas' and finding length to extract 9 digit number
    adm1_name = adm1[:-10]
    adm1_len = len(adm1_name)

    # Extracting 9 digit number
    adm1_num = adm1[adm1_len+1:]

    # need to include test for any instance where the length of the 9 digit number does not equal 9 (just incase)
    # adm1_num_len = len(adm1_num)
    # if adm1_num_len <> 9:
    dir_out = path_out / var / adm0 / crop
    path_outf = dir_out / Path(adm1_name + '_' + adm1_num + '_' + str(yr) + '_' + var + '_' + crop + '.csv')

    util.make_dir_if_missing(dir_out)

    # Process variable:
    # 1. if output csv does not exist OR
    # 2. if processing current year OR
    # 3. if REDO flag is set to true
    if not os.path.isfile(path_outf) or datetime.datetime.now().year == yr or constants.do_redo:
        with MemoryFile(open(crop_mask, 'rb').read()) as memfile:
            with memfile.open() as hndl_crop_mask:
                mask_crop_per = hndl_crop_mask.read(1).astype(float)

                if constants_base.do_threshold:
                    mask_crop_per[mask_crop_per < constants_base.lower_threshold] = 0.0  # Create crop mask and mask pixel LT CP

                    # If no pixels then reduce threshold by half
                    if not np.count_nonzero(mask_crop_per):
                        # logger.error('Reducing threshold by half for ' + adm1 + ' for crop ' + crop_name)
                        mask_crop_per = hndl_crop_mask.read(1).astype(float)
                        mask_crop_per[mask_crop_per < constants_base.lower_threshold/2.] = 0.0  # Create crop mask and mask pixel LT CP
                    # If no pixels then reduce threshold by half
                    if not np.count_nonzero(mask_crop_per):
                        # logger.error('Reducing threshold by half for ' + adm1 + ' for crop ' + crop_name)
                        mask_crop_per = hndl_crop_mask.read(1).astype(float)
                        mask_crop_per[mask_crop_per < constants_base.lower_threshold / 4.] = 0.0  # Create crop mask and mask pixel LT CP
                else:
                    # TODO # If no pixels then reduce threshold by half
                    val_percentile = np.percentile(mask_crop_per[mask_crop_per > 0.], constants_base.upper_percentile)
                    mask_crop_per[mask_crop_per < val_percentile] = 0.0

                if np.count_nonzero(mask_crop_per):  # if there are no pixels then skip
                    # print(adm0 + ' ' + adm1_name + ' ' + adm1_num + ' ' + crop + ' ' + var + ' ' + str(yr))
                    tmp_str = compute_stats(adm0, adm1_name, adm1_num, yr, var, mask_crop_per, path_outf)

                    # Append all strings together
                    data_out = '\n'.join(tmp_str)

                    hndl_out = open(path_outf, 'w')
                    hndl_out.write(data_out)
                    hndl_out.close()
    else:
        pass


def remove_duplicates(lst):
    return list(set([i for i in lst]))


def run(params):
    """

    Returns:

    """
    all_comb = []

    # Parse config file
    import ast
    pdb.set_trace()
    for adm0 in params.countries:
        for crop in ast.literal_eval(params.parser.get(adm0, 'crops')):
            name_crop = 'cr' if params.parser.getboolean(adm0, 'use_cropland_mask') else crop
            list_cmasks = glob.glob(str(path_cmasks) + os.sep + adm0 + os.sep + name_crop + os.sep + '*_' + name_crop + '_crop_mask.tif')

            if len(list_cmasks):
                for var in ast.literal_eval(params.parser.get(adm0, 'eo_model')):
                    if var in ['crop_stats', 'GDD']:
                        continue
                    all_comb.extend(list(itertools.product([adm0], [name_crop], [var], list_yrs, list(list_cmasks))))
                all_comb.extend(list(itertools.product([adm0], [name_crop], list_yrs, list(list_cmasks))))

    all_comb = remove_duplicates(all_comb)

    params.logger.error('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    params.logger.error('REDO flag: ' + str(constants.do_redo))
    params.logger.error('Number CPUs: ' + str(params.fraction_cpus))
    params.logger.error(list_countries)
    params.logger.error(list_crops)
    params.logger.error(list_vars)
    params.logger.error(str(syr) + ' ' + str(eyr))
    params.logger.error('Total number of csvs to process: ' + str(len(all_comb)))
    params.logger.error('Storing outputs at ' + str(path_out))
    params.logger.error('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    if False and constants.do_parallel:
        with Pool(params.fraction_cpus) as p:
            with tqdm(total=len(all_comb)) as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(process, all_comb))):
                    pbar.set_description('Processing ' + ' '.join(str(x) for x in all_comb[i][:4]) + ' ' + os.path.basename(all_comb[i][4]))
                    pbar.update()
    else:
        # Use the code below if you want to test without parallelization or if you want to debug by using pdb
        pbar = tqdm(all_comb)
        for i, val in enumerate(pbar):
            pbar.set_description('Processing ' + ' '.join(str(x) for x in all_comb[i][:4]) + ' ' + os.path.basename(all_comb[i][4]) + '\n')
            pbar.update()
            process(val)


if __name__ == '__main__':
    pass
