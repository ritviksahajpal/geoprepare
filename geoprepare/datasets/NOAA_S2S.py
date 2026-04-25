#!/usr/bin/env python3
"""
Download and process NOAA PSL S2S monthly-to-seasonal forecast data.

Provides seasonal precipitation (tprate, mm/day) and temperature (t2m, K→°C)
forecasts and hindcasts from 4 models: ECCC, ECMWF, NCEP, UKMO.

Data is pre-extracted to 1109 polygons (FNID) matching adm_shapefile.gpkg.

URLs:
  Forecasts: https://downloads.psl.noaa.gov/Projects/s2s_C3S_monthly_to_seasonal/data/forecasts/{var}/{MODEL}/
  Hindcasts: https://downloads.psl.noaa.gov/Projects/s2s_C3S_monthly_to_seasonal/data/hindcasts/{var}/{MODEL}/

NetCDF structure:
  Dimensions: number (ensemble), L (lead months), FNID (polygons), reference_time
  Variables: hindcasts/forecasts(FNID, number, reference_time, L)
  FNID: string identifier matching adm_shapefile.gpkg FNID column

Steps:
  1. Scrape directory listings and download .nc files from all models.
  2. Read NetCDF, match FNID to admin regions, compute ensemble mean+spread.
  3. Output CSVs per country with lead-time rows.
"""
import logging
import os
from pathlib import Path
from html.parser import HTMLParser

import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://downloads.psl.noaa.gov/Projects/s2s_C3S_monthly_to_seasonal/data"
MODELS = ["ECCC", "ECMWF", "NCEP", "UKMO"]
VARIABLES = ["t2m", "tprate"]
DATA_TYPES = ["forecasts", "hindcasts"]

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "geoprepare/NOAA-S2S"})


class _LinkParser(HTMLParser):
    """Extract .nc file links from an HTML directory listing."""

    def __init__(self):
        super().__init__()
        self.files = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value.endswith(".nc"):
                    self.files.append(value)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    reraise=True,
)
def _fetch(url, timeout=120):
    """Fetch URL content with retry."""
    resp = _SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def _list_nc_files(url):
    """Scrape a directory listing and return list of .nc filenames."""
    try:
        resp = _fetch(url)
        parser = _LinkParser()
        parser.feed(resp.text)
        return parser.files
    except Exception as e:
        logger.warning(f"Failed to list {url}: {e}")
        return []


def _download_file(url, out_path):
    """Download a single file if it doesn't exist."""
    if out_path.exists():
        return

    try:
        resp = _fetch(url, timeout=300)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if out_path.exists():
            out_path.unlink()


def download_all(dir_download):
    """Download all S2S forecast and hindcast files from all models."""
    base_dir = Path(dir_download) / "noaa_s2s"

    tasks = []
    for data_type in DATA_TYPES:
        for var in VARIABLES:
            for model in MODELS:
                url = f"{BASE_URL}/{data_type}/{var}/{model}/"
                out_dir = base_dir / data_type / var / model

                nc_files = _list_nc_files(url)
                for fname in nc_files:
                    tasks.append((f"{url}{fname}", out_dir / fname))

    logger.info(f"Downloading {len(tasks)} NOAA S2S files")
    for file_url, out_path in tqdm(tasks, desc="Download NOAA S2S"):
        _download_file(file_url, out_path)


def process_file(nc_path, fnid_to_info, var, data_type):
    """
    Read a single NetCDF file and return a list of row dicts.

    Each row: {country, admin1, admin2, fnid, model, init_month, year,
               lead, var_mean, var_spread}
    """
    try:
        import netCDF4 as nc
    except ImportError:
        import xarray as xr
        return _process_file_xarray(nc_path, fnid_to_info, var, data_type)

    rows = []
    try:
        ds = nc.Dataset(str(nc_path), "r")
    except Exception as e:
        logger.error(f"Failed to open {nc_path}: {e}")
        return rows

    try:
        # Read FNID strings
        fnid_var = ds.variables["FNID"]
        fnids = ["".join(f.decode() if isinstance(f, bytes) else f for f in row).strip()
                 for row in fnid_var[:]]

        # Read data variable (hindcasts or forecasts)
        data_var_name = data_type  # "hindcasts" or "forecasts"
        if data_var_name not in ds.variables:
            # Try alternative names
            for vname in ds.variables:
                if vname not in ("FNID", "number", "L", "reference_time"):
                    data_var_name = vname
                    break

        data = ds.variables[data_var_name][:]  # (FNID, number, reference_time, L)

        # Read reference_time (seconds since 1970-01-01)
        ref_times = ds.variables["reference_time"][:]

        # Read lead months
        leads = ds.variables["L"][:]

        # Extract model name from filename
        model_name = nc_path.stem.split(f"_{data_type}")[0]

        # Extract init month from filename
        # Hindcast: MODEL_hindcasts_MM_1993-2016_var.nc
        # Forecast: MODEL_forecasts_MMYYYY_var.nc
        parts = nc_path.stem.split(f"_{data_type}_")[1]
        if data_type == "hindcasts":
            init_month = int(parts.split("_")[0])
        else:
            init_month = int(parts[:2])

        # Process each FNID
        for i, fnid in enumerate(fnids):
            if fnid not in fnid_to_info:
                continue

            info = fnid_to_info[fnid]

            # data[i] shape: (number, reference_time, L)
            region_data = np.array(data[i])

            for t_idx, ref_time in enumerate(ref_times):
                # Convert reference_time to year
                from datetime import datetime, timezone
                year = datetime.fromtimestamp(
                    float(ref_time), tz=timezone.utc
                ).year

                for l_idx, lead in enumerate(leads):
                    # Slice: all ensemble members for this (ref_time, lead)
                    ensemble = region_data[:, t_idx, l_idx]

                    # Filter NaN
                    valid = ensemble[~np.isnan(ensemble)]
                    if len(valid) == 0:
                        continue

                    value_mean = float(np.mean(valid))
                    value_spread = float(np.std(valid))

                    # Convert temperature from Kelvin to Celsius
                    if var == "t2m":
                        value_mean -= 273.15
                        value_spread = value_spread  # std doesn't shift

                    rows.append({
                        "country": info["country"],
                        "admin1": info["admin1"],
                        "admin2": info["admin2"],
                        "fnid": fnid,
                        "model": model_name,
                        "init_month": init_month,
                        "year": year,
                        "lead": int(lead),
                        f"{var}_mean": round(value_mean, 4),
                        f"{var}_spread": round(value_spread, 4),
                    })
    except Exception as e:
        logger.error(f"Error processing {nc_path}: {e}")
    finally:
        ds.close()

    return rows


def _process_file_xarray(nc_path, fnid_to_info, var, data_type):
    """Fallback processing using xarray instead of netCDF4."""
    import xarray as xr

    rows = []
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        logger.error(f"Failed to open {nc_path}: {e}")
        return rows

    try:
        fnids = [str(f).strip() for f in ds["FNID"].values]

        # Find the data variable
        data_var_name = data_type
        if data_var_name not in ds:
            for vname in ds.data_vars:
                if vname != "FNID":
                    data_var_name = vname
                    break

        data = ds[data_var_name].values
        ref_times = ds["reference_time"].values
        leads = ds["L"].values

        model_name = nc_path.stem.split(f"_{data_type}")[0]
        parts = nc_path.stem.split(f"_{data_type}_")[1]
        if data_type == "hindcasts":
            init_month = int(parts.split("_")[0])
        else:
            init_month = int(parts[:2])

        for i, fnid in enumerate(fnids):
            if fnid not in fnid_to_info:
                continue

            info = fnid_to_info[fnid]
            region_data = data[i]

            for t_idx, ref_time in enumerate(ref_times):
                import pandas as pd
                year = pd.Timestamp(ref_time).year

                for l_idx, lead in enumerate(leads):
                    ensemble = region_data[:, t_idx, l_idx]
                    valid = ensemble[~np.isnan(ensemble)]
                    if len(valid) == 0:
                        continue

                    value_mean = float(np.mean(valid))
                    value_spread = float(np.std(valid))

                    if var == "t2m":
                        value_mean -= 273.15

                    rows.append({
                        "country": info["country"],
                        "admin1": info["admin1"],
                        "admin2": info["admin2"],
                        "fnid": fnid,
                        "model": model_name,
                        "init_month": init_month,
                        "year": year,
                        "lead": int(lead),
                        f"{var}_mean": round(value_mean, 4),
                        f"{var}_spread": round(value_spread, 4),
                    })
    except Exception as e:
        logger.error(f"Error processing {nc_path}: {e}")
    finally:
        ds.close()

    return rows


def _load_fnid_mapping(shapefile_path):
    """Load FNID → region info mapping from the boundary shapefile."""
    import geopandas as gp

    gdf = gp.read_file(shapefile_path)
    fnid_to_info = {}
    for _, row in gdf.iterrows():
        fnid_to_info[row["FNID"]] = {
            "country": row["ADMIN0"],
            "admin1": row.get("ADMIN1", ""),
            "admin2": row.get("ADMIN2", "") or "",
        }
    return fnid_to_info


def process_all(dir_download, dir_output, shapefile_path, countries=None):
    """
    Process all downloaded S2S NetCDF files and output CSVs per country.

    Output: {dir_output}/noaa_s2s/{country}/{var}_{data_type}.csv
    """
    import pandas as pd

    base_dir = Path(dir_download) / "noaa_s2s"
    out_base = Path(dir_output) / "noaa_s2s"

    fnid_to_info = _load_fnid_mapping(shapefile_path)
    logger.info(f"Loaded {len(fnid_to_info)} FNID mappings")

    for data_type in DATA_TYPES:
        for var in VARIABLES:
            all_rows = []

            # Collect all .nc files for this data_type/var across all models
            nc_files = []
            for model in MODELS:
                model_dir = base_dir / data_type / var / model
                if model_dir.exists():
                    nc_files.extend(sorted(model_dir.glob("*.nc")))

            if not nc_files:
                logger.warning(f"No files found for {data_type}/{var}")
                continue

            logger.info(f"Processing {len(nc_files)} files for {data_type}/{var}")
            for nc_path in tqdm(nc_files, desc=f"Process {data_type}/{var}"):
                rows = process_file(nc_path, fnid_to_info, var, data_type)
                all_rows.extend(rows)

            if not all_rows:
                continue

            df = pd.DataFrame(all_rows)

            # Write per-country CSVs
            for country, df_country in df.groupby("country"):
                country_slug = country.lower().replace(" ", "_")

                # Filter by requested countries if specified
                if countries and country_slug not in [
                    c.lower().replace(" ", "_") for c in countries
                ]:
                    continue

                out_dir = out_base / country_slug
                out_dir.mkdir(parents=True, exist_ok=True)

                out_file = out_dir / f"{var}_{data_type}.csv"
                df_country.to_csv(out_file, index=False)
                logger.info(
                    f"  {country}: {len(df_country)} rows → {out_file}"
                )


def run(geoprep):
    """Main entry point for NOAA S2S processing.

    Args:
        geoprep: GeoDownload object with attributes:
            dir_download, dir_output, dir_boundary_files,
            countries (optional).
    """
    dir_download = Path(geoprep.dir_download)
    dir_output = Path(geoprep.dir_output)

    # Shapefile for FNID mapping
    shapefile_path = (
        Path(geoprep.dir_boundary_files) / "adm_shapefile.gpkg"
    )

    countries = getattr(geoprep, "countries", None)

    # Step 1: Download
    download_all(dir_download)

    # Step 2: Process and output CSVs
    process_all(dir_download, dir_output, shapefile_path, countries)
