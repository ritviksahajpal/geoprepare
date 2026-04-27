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

        # Read reference_time if it exists (hindcasts have 24 years, forecasts may have 1 or none)
        if "reference_time" in ds.variables:
            ref_times = ds.variables["reference_time"][:]
        else:
            ref_times = None

        # Read lead months
        leads = ds.variables["L"][:]

        # Extract model name from filename
        model_name = nc_path.stem.split(f"_{data_type}")[0]

        # Extract init month and year from filename
        # Hindcast: MODEL_hindcasts_MM_1993-2016_var.nc
        # Forecast: MODEL_forecasts_MMYYYY_var.nc
        parts = nc_path.stem.split(f"_{data_type}_")[1]
        if data_type == "hindcasts":
            init_month = int(parts.split("_")[0])
        else:
            init_month = int(parts[:2])

        # Build list of (ref_time_index, year) pairs
        from datetime import datetime, timezone
        time_entries = []
        if ref_times is not None:
            for t_idx, ref_time in enumerate(ref_times):
                try:
                    rt = float(ref_time)
                    if np.isnan(rt):
                        continue
                    year = datetime.fromtimestamp(rt, tz=timezone.utc).year
                    time_entries.append((t_idx, year))
                except (ValueError, OSError):
                    continue
        else:
            # Forecast files without reference_time: extract year from filename
            forecast_year = int(parts[2:6]) if len(parts) >= 6 else datetime.now().year
            time_entries.append((0, forecast_year))

        # Process each FNID
        for i, fnid in enumerate(fnids):
            if fnid not in fnid_to_info:
                continue

            info = fnid_to_info[fnid]

            # data[i] shape varies: (number, L) or (number, reference_time, L)
            region_data = np.array(data[i])

            for t_idx, year in time_entries:
                for l_idx, lead in enumerate(leads):
                    # Handle different array shapes
                    if region_data.ndim == 3:
                        # (number, reference_time, L)
                        ensemble = region_data[:, t_idx, l_idx]
                    elif region_data.ndim == 2:
                        # (number, L) — no reference_time dimension
                        ensemble = region_data[:, l_idx]
                    else:
                        continue

                    # Filter NaN
                    valid = ensemble[~np.isnan(ensemble)]
                    if len(valid) == 0:
                        continue

                    value_mean = float(np.mean(valid))
                    value_spread = float(np.std(valid))

                    # Convert temperature from Kelvin to Celsius
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
        ref_times = ds["reference_time"].values if "reference_time" in ds else None
        leads = ds["L"].values

        model_name = nc_path.stem.split(f"_{data_type}")[0]
        parts = nc_path.stem.split(f"_{data_type}_")[1]
        if data_type == "hindcasts":
            init_month = int(parts.split("_")[0])
        else:
            init_month = int(parts[:2])

        # Build time entries
        import pandas as pd
        from datetime import datetime
        time_entries = []
        if ref_times is not None:
            for t_idx, ref_time in enumerate(ref_times):
                try:
                    year = pd.Timestamp(ref_time).year
                    time_entries.append((t_idx, year))
                except Exception:
                    continue
        else:
            forecast_year = int(parts[2:6]) if len(parts) >= 6 else datetime.now().year
            time_entries.append((0, forecast_year))

        for i, fnid in enumerate(fnids):
            if fnid not in fnid_to_info:
                continue

            info = fnid_to_info[fnid]
            region_data = data[i]

            for t_idx, year in time_entries:
                for l_idx, lead in enumerate(leads):
                    if region_data.ndim == 3:
                        ensemble = region_data[:, t_idx, l_idx]
                    elif region_data.ndim == 2:
                        ensemble = region_data[:, l_idx]
                    else:
                        continue

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
        admin2 = row.get("ADMIN2", "")
        fnid_to_info[row["FNID"]] = {
            "country": row["ADMIN0"],
            "admin1": row.get("ADMIN1", "") or "",
            "admin2": admin2 if isinstance(admin2, str) else "",
        }
    return fnid_to_info


S2S_NUM_LEADS = 6
S2S_VAR_PREFIX = "s2s"  # output columns: s2s_t2m_lead1, s2s_tprate_lead1, etc.


def process_all(dir_download, dir_output, shapefile_path, countries=None,
                scale="admin_1", crop="cr", threshold_dir="crop_t20",
                gap_fill="climatology"):
    """
    Process all downloaded S2S NetCDF files and output per-region CSVs
    matching the FLDAS extraction output format.

    Output per region/year CSV at:
      {dir_output}/{threshold_dir}/{country}/{scale}/{crop}/s2s_{var}/
          {fnid}_{region}_{year}_s2s_{var}_{crop}.csv

    CSV columns: country, region, region_id, year, month,
                 s2s_{var}_lead1, ..., s2s_{var}_lead6
    """
    import pandas as pd

    base_dir = Path(dir_download) / "noaa_s2s"

    fnid_to_info = _load_fnid_mapping(shapefile_path)
    logger.info(f"Loaded {len(fnid_to_info)} FNID mappings")

    # Process each variable: collect hindcast + forecast rows, then gap-fill
    for var in VARIABLES:
        s2s_var = f"{S2S_VAR_PREFIX}_{var}"
        mean_col = f"{var}_mean"
        all_rows = []

        # Process both hindcasts and forecasts for this variable
        for data_type in DATA_TYPES:
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
        logger.info(f"  {var}: {len(df)} total rows from hindcasts + forecasts")

        # Fill NaN admin2 to avoid groupby dropping rows with NaN keys
        df["admin2"] = df["admin2"].fillna("")

        # Compute multi-model mean across models per (fnid, init_month, year, lead)
        group_cols = ["country", "admin1", "admin2", "fnid", "init_month", "year", "lead"]
        df_mm = df.groupby(group_cols, as_index=False)[mean_col].mean()

        # Pivot leads into columns: s2s_{var}_lead1 ... s2s_{var}_lead6
        df_mm["lead_col"] = df_mm["lead"].apply(lambda x: f"{s2s_var}_lead{x}")
        pivot_cols = ["country", "admin1", "admin2", "fnid", "init_month", "year"]
        df_pivot = df_mm.pivot_table(
            index=pivot_cols, columns="lead_col", values=mean_col
        ).reset_index()
        df_pivot.columns.name = None

        # Rename init_month → month to match FLDAS format
        df_pivot.rename(columns={"init_month": "month"}, inplace=True)

        # Ensure all lead columns exist (lead1..lead6)
        lead_cols = [f"{s2s_var}_lead{i}" for i in range(1, S2S_NUM_LEADS + 1)]
        for col in lead_cols:
            if col not in df_pivot.columns:
                df_pivot[col] = float("nan")

        # Gap-fill years between hindcast end and current year with climatology.
        # Only the current year uses actual forecast data; all prior years
        # after hindcast end (2017 through current_year-1) use climatology.
        from datetime import datetime as _dt
        current_year = _dt.now().year

        if gap_fill == "climatology" and not df_pivot.empty:
            hindcast_years = [y for y in df_pivot["year"].unique() if y <= 2016]
            if hindcast_years:
                hindcast_end = max(hindcast_years)
                gap_years = list(range(hindcast_end + 1, current_year))

                # Remove any incomplete forecast data for gap years
                # (e.g., 2025 may have only months 10-12 from forecasts)
                df_pivot = df_pivot[
                    (df_pivot["year"] <= hindcast_end) | (df_pivot["year"] >= current_year)
                ]

                if gap_years:
                    # Compute climatology: mean per (country, admin1, admin2, fnid, month) across hindcast years
                    hindcast_data = df_pivot[df_pivot["year"] <= hindcast_end]
                    clim_cols = ["country", "admin1", "admin2", "fnid", "month"]
                    climatology = hindcast_data.groupby(clim_cols, as_index=False)[lead_cols].mean()

                    gap_frames = []
                    for gap_year in gap_years:
                        clim_copy = climatology.copy()
                        clim_copy["year"] = gap_year
                        gap_frames.append(clim_copy)

                    df_gap = pd.concat(gap_frames, ignore_index=True)
                    df_pivot = pd.concat([df_pivot, df_gap], ignore_index=True)
                    logger.info(f"  {var}: gap-filled {len(gap_years)} years ({gap_years[0]}-{gap_years[-1]}) with hindcast climatology")

        # Write per-region/year CSVs matching extraction output structure
        for (country, fnid), df_region in df_pivot.groupby(["country", "fnid"]):
            country_slug = country.lower().replace(" ", "_")

            if countries and country_slug not in [
                c.lower().replace(" ", "_") for c in countries
            ]:
                continue

            info = fnid_to_info.get(fnid, {})
            admin1 = (info.get("admin1", "") or "").lower().replace(" ", "_")
            admin2 = (info.get("admin2", "") or "").lower().replace(" ", "_")
            region = admin2 if admin2 and scale == "admin_2" else admin1

            # Output directory matching geoextract structure
            out_dir = (
                Path(dir_output) / threshold_dir / country_slug
                / scale / crop / s2s_var
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            for year, df_year in df_region.groupby("year"):
                out_file = out_dir / f"{fnid}_{region}_{int(year)}_{s2s_var}_{crop}.csv"

                df_out = pd.DataFrame({
                    "country": country_slug,
                    "region": region,
                    "region_id": fnid,
                    "year": int(year),
                    "month": df_year["month"].values,
                })
                for col in lead_cols:
                    df_out[col] = df_year[col].values

                df_out.to_csv(out_file, index=False)

        logger.info(f"  {var}: wrote per-region CSVs")


def run(geoprep):
    """Main entry point for NOAA S2S processing.

    Args:
        geoprep: GeoDownload object with attributes:
            dir_download, dir_output, dir_boundary_files,
            s2s_countries, s2s_scale, s2s_threshold_dir, s2s_crop.
    """
    dir_download = Path(geoprep.dir_download)
    dir_output = Path(geoprep.dir_output)

    # Shapefile for FNID mapping
    shapefile_path = (
        Path(geoprep.dir_boundary_files) / "adm_shapefile.gpkg"
    )

    countries = getattr(geoprep, "s2s_countries", None)
    scale = getattr(geoprep, "s2s_scale", "admin_1")
    threshold_dir = getattr(geoprep, "s2s_threshold_dir", "crop_t20")
    crop = getattr(geoprep, "s2s_crop", "cr")
    project_name = getattr(geoprep, "s2s_project_name", "")
    dir_output = dir_output / project_name

    # Step 1: Download
    download_all(dir_download)

    gap_fill = getattr(geoprep, "s2s_gap_fill", "climatology")

    # Step 2: Process and output per-region CSVs
    process_all(
        dir_download, dir_output, shapefile_path, countries,
        scale=scale, crop=crop, threshold_dir=threshold_dir,
        gap_fill=gap_fill,
    )
