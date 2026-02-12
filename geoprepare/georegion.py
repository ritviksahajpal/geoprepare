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
#
# Plotting:
#   plot_region_assignments()  — single country, one subplot per region
#   plot_all_countries()       — batch all EWCM countries from config
###############################################################################
import os
import ast
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


def _admin_name_col(scale, columns):
    """Choose ADMIN2 or ADMIN1 column based on scale and availability."""
    if scale == "admin_2" and ADM_ADMIN2_COL in columns:
        return ADM_ADMIN2_COL
    return ADM_ADMIN1_COL


# ---------------------------------------------------------------------------
#  Shared data loading
# ---------------------------------------------------------------------------
def _load_country_data(path_admin_shp, path_region_shp, country):
    """
    Load and filter admin and region shapefiles for a single country.

    Returns
    -------
    gdf_admin : GeoDataFrame
        Admin units filtered to the country, with standardised column names.
    gdf_region : GeoDataFrame
        EWCM regions filtered to the country.

    Both are CRS-aligned (admin reprojected to region CRS if they differ).
    """
    gdf_admin = gp.read_file(Path(path_admin_shp), engine="pyogrio")
    gdf_region = gp.read_file(Path(path_region_shp), engine="pyogrio")

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

    # Filter to the requested country
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

    # Align CRS (reproject admin to region CRS if they differ)
    if gdf_admin.crs != gdf_region.crs:
        log.info(
            f"Reprojecting admin CRS {gdf_admin.crs} → {gdf_region.crs}"
        )
        gdf_admin = gdf_admin.to_crs(gdf_region.crs)

    return gdf_admin, gdf_region


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
    # 2. Load and filter shapefiles
    # ------------------------------------------------------------------
    gdf_admin, gdf_region = _load_country_data(
        path_admin_shp, path_region_shp, country
    )

    # ------------------------------------------------------------------
    # 3. Project to equal-area CRS for accurate area calculations
    # ------------------------------------------------------------------
    ea_crs = "EPSG:6933"
    gdf_admin_ea = gdf_admin.to_crs(ea_crs)
    gdf_region_ea = gdf_region.to_crs(ea_crs)

    # Pre-compute admin unit areas
    gdf_admin_ea["_admin_area"] = gdf_admin_ea.geometry.area

    # ------------------------------------------------------------------
    # 4. Overlay (intersection) — produces one row per admin×region pair
    #    that actually overlaps
    # ------------------------------------------------------------------
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
    # 5. Compute intersection area and pick largest overlap per admin unit
    # ------------------------------------------------------------------
    gdf_overlay["_overlap_area"] = gdf_overlay.geometry.area
    gdf_overlay["_overlap_pct"] = (
        gdf_overlay["_overlap_area"] / gdf_overlay["_admin_area"]
    )

    idx_max = gdf_overlay.groupby(ADM_ID_COL)["_overlap_area"].idxmax()
    df_best = gdf_overlay.loc[idx_max].copy()

    # ------------------------------------------------------------------
    # 6. Choose admin name column based on scale
    # ------------------------------------------------------------------
    name_col = _admin_name_col(scale, df_best.columns)

    # ------------------------------------------------------------------
    # 7. Build clean output
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

    result = result.sort_values("admin_name").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8. Log diagnostics
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
    # 9. Cache result
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


# ---------------------------------------------------------------------------
#  Plotting: one subplot per region showing its assigned admin zones
# ---------------------------------------------------------------------------
def plot_region_assignments(
    path_admin_shp,
    path_region_shp,
    country,
    scale="admin_1",
    dir_cache=None,
    path_output=None,
    ncols=3,
    figsize_per_subplot=(5, 5),
    label_fontsize=5,
    title_fontsize=10,
):
    """
    Plot a multi-panel figure: one subplot per EWCM region showing the
    admin zones assigned to it.

    Each subplot contains:
    - The EWCM region boundary as a bold black outline
    - Admin zones filled with distinct colors, white borders between them
    - Admin zone names labelled at their representative point

    Parameters
    ----------
    path_admin_shp, path_region_shp, country, scale, dir_cache
        Same as assign_admin_to_regions.
    path_output : str or Path, optional
        If provided, save the figure to this path (e.g. PNG or PDF).
    ncols : int
        Number of columns in the subplot grid.
    figsize_per_subplot : tuple of (width, height)
        Size of each subplot in inches.
    label_fontsize : int
        Font size for admin zone labels.
    title_fontsize : int
        Font size for subplot titles.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # Get assignment mapping (may hit cache)
    df_assignment = assign_admin_to_regions(
        path_admin_shp, path_region_shp, country, scale, dir_cache
    )

    # Load geometries for plotting
    gdf_admin, gdf_region = _load_country_data(
        path_admin_shp, path_region_shp, country
    )

    # Add normalized admin name for joining
    name_col = _admin_name_col(scale, gdf_admin.columns)
    gdf_admin["_admin_name"] = gdf_admin[name_col].apply(_normalize)

    # Join region assignment onto admin geometries
    gdf_admin = gdf_admin.merge(
        df_assignment[["admin_name", "region_name"]],
        left_on="_admin_name",
        right_on="admin_name",
        how="left",
    )

    # Determine subplot grid
    regions = sorted(gdf_admin["region_name"].dropna().unique())
    n_regions = len(regions)

    if n_regions == 0:
        log.warning(f"No regions found for '{country}', nothing to plot.")
        return None

    nrows = -(-n_regions // ncols)  # ceil division
    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=False
    )
    axes_flat = axes.flatten()

    cmap = plt.colormaps["tab20"]

    for i, region in enumerate(regions):
        ax = axes_flat[i]

        # Admin zones assigned to this region
        gdf_sub = gdf_admin[gdf_admin["region_name"] == region].copy()

        # EWCM region boundary
        gdf_reg = gdf_region[
            gdf_region[REG_NAME_COL].apply(_normalize) == _normalize(region)
        ]

        # Color each admin zone distinctly
        n_admins = len(gdf_sub)
        colors = [cmap(j % 20) for j in range(n_admins)]
        gdf_sub.plot(
            ax=ax, color=colors, edgecolor="white", linewidth=0.8, alpha=0.7
        )

        # Bold region boundary on top
        gdf_reg.boundary.plot(ax=ax, color="black", linewidth=2.0)

        # Label each admin zone at its representative point
        # (representative_point is guaranteed inside the polygon, unlike centroid)
        for _, row in gdf_sub.iterrows():
            pt = row.geometry.representative_point()
            label = row["_admin_name"].replace("_", " ").title()
            ax.annotate(
                label,
                xy=(pt.x, pt.y),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                fontweight="medium",
            )

        ax.set_title(region, fontsize=title_fontsize, fontweight="bold")
        ax.set_axis_off()

    # Hide unused subplot slots
    for j in range(n_regions, len(axes_flat)):
        axes_flat[j].set_visible(False)

    country_title = country.replace("_", " ").title()
    fig.suptitle(
        f"EWCM Region \u2192 Admin Zone Assignments: {country_title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if path_output:
        path_output = Path(path_output)
        os.makedirs(path_output.parent, exist_ok=True)
        fig.savefig(path_output, dpi=200, bbox_inches="tight")
        log.info(f"Saved region assignment plot: {path_output}")

    return fig


def plot_all_countries(
    path_config_file,
    path_admin_shp,
    path_region_shp,
    dir_output,
    dir_cache=None,
    country_default="malawi",
    **plot_kwargs,
):
    """
    Generate region-assignment plots for all EWCM countries in the config.

    Reads country names and scales from the config file (e.g. geoextract.txt).
    Only countries with ``category = EWCM`` are included.

    Parameters
    ----------
    path_config_file : str or list of str
        Path(s) to the config file(s) (e.g. ["geobase.txt", "geoextract.txt"]).
    path_admin_shp : str or Path
        Path to the admin-level shapefile.
    path_region_shp : str or Path
        Path to the EWCM regions shapefile.
    dir_output : str or Path
        Directory where per-country PNGs are saved.
    dir_cache : str or Path, optional
        Cache directory for region lookups.
    country_default : str
        Country processed first (useful for quick sanity-check).
    **plot_kwargs
        Extra keyword arguments forwarded to plot_region_assignments
        (e.g. ncols, figsize_per_subplot, label_fontsize).

    Returns
    -------
    dict
        {country: Path} mapping each country to its saved plot path.
        Countries that failed are logged and omitted from the dict.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for batch runs
    import matplotlib.pyplot as plt

    from . import utils

    dir_output = Path(dir_output)
    os.makedirs(dir_output, exist_ok=True)

    parser = utils.read_config(path_config_file)

    # Collect EWCM countries and their scales from config
    ewcm_countries = {}
    for section in parser.sections():
        try:
            category = parser.get(section, "category")
        except Exception:
            continue
        if _normalize(category) != "ewcm":
            continue
        # Read the scale; default to admin_1
        try:
            scales = ast.literal_eval(parser.get(section, "scales"))
            scale = scales[0] if scales else "admin_1"
        except Exception:
            scale = "admin_1"
        ewcm_countries[section] = scale

    if not ewcm_countries:
        log.warning("No EWCM countries found in config.")
        return {}

    # Process default country first for quick visual sanity-check
    ordered = []
    if country_default in ewcm_countries:
        ordered.append(country_default)
    ordered.extend(c for c in sorted(ewcm_countries) if c != country_default)

    results = {}
    for country in ordered:
        scale = ewcm_countries[country]
        out_path = dir_output / f"region_assignments_{country}.png"
        log.info(f"Plotting {country} (scale={scale}) → {out_path}")
        try:
            fig = plot_region_assignments(
                path_admin_shp=path_admin_shp,
                path_region_shp=path_region_shp,
                country=country,
                scale=scale,
                dir_cache=dir_cache,
                path_output=out_path,
                **plot_kwargs,
            )
            if fig is not None:
                plt.close(fig)  # free memory in batch mode
            results[country] = out_path
        except Exception as exc:
            log.error(f"Failed to plot '{country}': {exc}")

    log.info(
        f"Completed {len(results)}/{len(ewcm_countries)} country plots "
        f"in {dir_output}"
    )
    return results