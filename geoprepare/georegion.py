###############################################################################
# Region Assignment: Map admin units to EWCM regions by largest spatial overlap
#
# Each admin unit (from adm_shapefile.shp) is assigned to exactly one
# EWCM region (from EWCM_Regions_v39.shp) — the region with which it
# shares the largest intersection area.
#
# Integration with geomerge.py:
#   Called before add_calendar() so that each admin unit's calendar_region
#   is populated from the spatial lookup rather than requiring it in the
#   statistics CSV.
#
#   from . import georegion
#   lookup = georegion.get_region_lookup(path_admin, path_region, country)
#   df["calendar_region"] = df["region"].map(lookup)
###############################################################################
import os
import hashlib
import logging

import numpy as np
import pandas as pd
import geopandas as gp
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Column names (match your shapefiles)
# ---------------------------------------------------------------------------
# adm_shapefile.shp
ADM_COUNTRY_COL = "ADMIN0"
ADM_ADMIN1_COL = "ADMIN1"
ADM_ADMIN2_COL = "ADMIN2"
ADM_ID_COL = "FNID"

# EWCM_Regions_v39.shp
REG_COUNTRY_COL = "ADM0_NAME"
REG_NAME_COL = "Name"
REG_KEY_COL = "Key"
REG_KEY2_COL = "Key2"


def _normalize(s):
    """Lowercase, strip, replace spaces with underscores."""
    if pd.isna(s):
        return s
    return str(s).strip().lower().replace(" ", "_")


def _cache_path(dir_cache, country, admin_hash, region_hash):
    """Build a deterministic cache filename."""
    tag = f"{country}_{admin_hash[:8]}_{region_hash[:8]}"
    return Path(dir_cache) / f"region_lookup_{tag}.csv"


def _file_hash(path):
    """Quick hash of file modification time + size for cache invalidation."""
    stat = os.stat(path)
    key = f"{path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
#  Core function
# ---------------------------------------------------------------------------
def assign_admin_to_regions(
    path_admin_shp,
    path_region_shp,
    country,
    scale="admin_1",
    dir_cache=None,
):
    """
    Assign each admin unit to the EWCM region with the largest overlap area.

    Parameters
    ----------
    path_admin_shp : str or Path
        Path to the admin-level shapefile (e.g. adm_shapefile.shp).
    path_region_shp : str or Path
        Path to the EWCM regions shapefile (e.g. EWCM_Regions_v39.shp).
    country : str
        Country name used to filter both shapefiles (case-insensitive).
    scale : str
        'admin_1' or 'admin_2' — determines which admin column is used as
        the unit name.
    dir_cache : str or Path, optional
        Directory to cache the lookup CSV.  If provided and a valid cache
        exists, the spatial overlay is skipped.

    Returns
    -------
    pd.DataFrame
        Columns:
            admin_id        – FNID from the admin shapefile
            admin_name      – ADMIN1 or ADMIN2 (lowercased, underscored)
            region_name     – EWCM region Name
            region_key      – EWCM region Key
            region_key2     – EWCM region Key2
            overlap_area    – intersection area in the overlay CRS
            overlap_pct     – intersection area / admin unit area
    """
    path_admin_shp = Path(path_admin_shp)
    path_region_shp = Path(path_region_shp)

    # ------------------------------------------------------------------
    # 1. Check cache
    # ------------------------------------------------------------------
    if dir_cache is not None:
        dir_cache = Path(dir_cache)
        os.makedirs(dir_cache, exist_ok=True)
        ah = _file_hash(path_admin_shp)
        rh = _file_hash(path_region_shp)
        cached = _cache_path(dir_cache, _normalize(country), ah, rh)
        if cached.is_file():
            log.info(f"Loading cached region lookup: {cached}")
            return pd.read_csv(cached)

    # ------------------------------------------------------------------
    # 2. Load shapefiles
    # ------------------------------------------------------------------
    gdf_admin = gp.read_file(path_admin_shp, engine="pyogrio")
    gdf_region = gp.read_file(path_region_shp, engine="pyogrio")

    # Standardise column names coming from different shapefile variants
    gdf_admin.rename(
        columns={
            "ADMIN0": ADM_COUNTRY_COL,
            "ADMIN1": ADM_ADMIN1_COL,
            "ADMIN2": ADM_ADMIN2_COL,
            "FNID": ADM_ID_COL,
            "name0": ADM_COUNTRY_COL,
            "name1": ADM_ADMIN1_COL,
            "asap1_id": ADM_ID_COL,
        },
        inplace=True,
    )

    # ------------------------------------------------------------------
    # 3. Filter to the requested country
    # ------------------------------------------------------------------
    country_norm = _normalize(country)

    mask_admin = gdf_admin[ADM_COUNTRY_COL].apply(_normalize) == country_norm
    gdf_admin = gdf_admin[mask_admin].copy()

    mask_region = gdf_region[REG_COUNTRY_COL].apply(_normalize) == country_norm
    gdf_region = gdf_region[mask_region].copy()

    if gdf_admin.empty:
        raise ValueError(
            f"No admin units found for country '{country}' in {path_admin_shp}"
        )
    if gdf_region.empty:
        raise ValueError(
            f"No regions found for country '{country}' in {path_region_shp}"
        )

    log.info(
        f"{country}: {len(gdf_admin)} admin units, {len(gdf_region)} regions"
    )

    # ------------------------------------------------------------------
    # 4. Align CRS (reproject admin to region CRS if they differ)
    # ------------------------------------------------------------------
    if gdf_admin.crs != gdf_region.crs:
        log.info(
            f"Reprojecting admin CRS {gdf_admin.crs} → {gdf_region.crs}"
        )
        gdf_admin = gdf_admin.to_crs(gdf_region.crs)

    # ------------------------------------------------------------------
    # 5. Project to equal-area CRS for accurate area calculations
    # ------------------------------------------------------------------
    # Use EPSG:6933 (World Cylindrical Equal Area) for area computations.
    ea_crs = "EPSG:6933"
    gdf_admin_ea = gdf_admin.to_crs(ea_crs)
    gdf_region_ea = gdf_region.to_crs(ea_crs)

    # Pre-compute admin unit areas
    gdf_admin_ea["_admin_area"] = gdf_admin_ea.geometry.area

    # ------------------------------------------------------------------
    # 6. Overlay (intersection) — produces one row per admin×region pair
    #    that actually overlaps
    # ------------------------------------------------------------------
    # Keep only the columns we need to reduce memory
    admin_cols = [ADM_ID_COL, ADM_ADMIN1_COL]
    if ADM_ADMIN2_COL in gdf_admin_ea.columns:
        admin_cols.append(ADM_ADMIN2_COL)
    admin_cols.append("_admin_area")

    region_cols = [REG_NAME_COL, REG_KEY_COL]
    if REG_KEY2_COL in gdf_region_ea.columns:
        region_cols.append(REG_KEY2_COL)

    gdf_overlay = gp.overlay(
        gdf_admin_ea[admin_cols + ["geometry"]],
        gdf_region_ea[region_cols + ["geometry"]],
        how="intersection",
        keep_geom_type=False,
    )

    if gdf_overlay.empty:
        raise ValueError(
            f"No spatial overlap between admin units and regions for '{country}'"
        )

    # ------------------------------------------------------------------
    # 7. Compute intersection area and pick largest overlap per admin unit
    # ------------------------------------------------------------------
    gdf_overlay["_overlap_area"] = gdf_overlay.geometry.area
    gdf_overlay["_overlap_pct"] = (
        gdf_overlay["_overlap_area"] / gdf_overlay["_admin_area"]
    )

    # For each admin unit keep only the region with the largest overlap
    idx_max = gdf_overlay.groupby(ADM_ID_COL)["_overlap_area"].idxmax()
    df_best = gdf_overlay.loc[idx_max].copy()

    # ------------------------------------------------------------------
    # 8. Choose admin name column based on scale
    # ------------------------------------------------------------------
    if scale == "admin_2" and ADM_ADMIN2_COL in df_best.columns:
        name_col = ADM_ADMIN2_COL
    else:
        name_col = ADM_ADMIN1_COL

    # ------------------------------------------------------------------
    # 9. Build clean output
    # ------------------------------------------------------------------
    result = pd.DataFrame(
        {
            "admin_id": df_best[ADM_ID_COL].values,
            "admin_name": df_best[name_col].apply(_normalize).values,
            "region_name": df_best[REG_NAME_COL].values,
            "region_key": df_best[REG_KEY_COL].values,
            "region_key2": (
                df_best[REG_KEY2_COL].values
                if REG_KEY2_COL in df_best.columns
                else np.nan
            ),
            "overlap_area": df_best["_overlap_area"].values,
            "overlap_pct": df_best["_overlap_pct"].values,
        }
    )

    # Sort by admin_name for readability
    result = result.sort_values("admin_name").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 10. Log diagnostics
    # ------------------------------------------------------------------
    n_total = len(gdf_admin)
    n_matched = len(result)
    n_unmatched = n_total - n_matched

    if n_unmatched > 0:
        matched_ids = set(result["admin_id"])
        missing = gdf_admin[~gdf_admin[ADM_ID_COL].isin(matched_ids)]
        log.warning(
            f"{n_unmatched}/{n_total} admin units had no overlap with any "
            f"region: {missing[ADM_ID_COL].tolist()}"
        )

    low_overlap = result[result["overlap_pct"] < 0.5]
    if not low_overlap.empty:
        log.warning(
            f"{len(low_overlap)} admin units have <50% overlap with their "
            f"assigned region:\n{low_overlap[['admin_id', 'admin_name', 'region_name', 'overlap_pct']].to_string()}"
        )

    log.info(f"Assigned {n_matched}/{n_total} admin units to regions")

    # ------------------------------------------------------------------
    # 11. Cache result
    # ------------------------------------------------------------------
    if dir_cache is not None:
        result.to_csv(cached, index=False)
        log.info(f"Cached region lookup: {cached}")

    return result


# ---------------------------------------------------------------------------
#  Convenience: get lookup as a dictionary  admin_name → region_name
# ---------------------------------------------------------------------------
def get_region_lookup(
    path_admin_shp,
    path_region_shp,
    country,
    scale="admin_1",
    dir_cache=None,
):
    """
    Return a dict mapping admin_name → region_name (calendar_region).

    Suitable for direct use in geomerge.py:
        lookup = georegion.get_region_lookup(...)
        df["calendar_region"] = df["region"].map(lookup)
    """
    df = assign_admin_to_regions(
        path_admin_shp, path_region_shp, country, scale, dir_cache
    )
    return dict(zip(df["admin_name"], df["region_name"].apply(_normalize)))