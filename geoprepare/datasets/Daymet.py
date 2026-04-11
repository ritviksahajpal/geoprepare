"""
Daymet V4 daily weather downloader for geoprepare.

Daymet V4 is daily gridded weather at 1 km native resolution covering
Continental North America (US, Canada, Mexico), Hawaii, and Puerto Rico,
1980-present (Puerto Rico: 1950-present). Distributed by ORNL DAAC via
NASA Earthdata OPeNDAP.

This module:
  1. Authenticates with NASA Earthdata Login (via earthaccess + ~/.netrc)
  2. Discovers annual OPeNDAP granules via the CMR
  3. Streams each granule with xarray+pydap and reprojects to 0.05 deg
     EPSG:4326 inside the user-supplied bbox
  4. Writes one GeoTIFF per day per variable

Variables and units (native):
  tmin / tmax : degrees C
  prcp        : mm/day
  vp          : Pa
  srad        : W/m2
  swe         : kg/m2
  dayl        : seconds/day

Leap years: Daymet uses a noleap 365-day calendar, so Dec 31 is missing on
leap years. We duplicate DOY 365 -> DOY 366 to match the rest of the
geoprepare pipeline which expects 366 days on leap years.

Prerequisites:
  pip install "geoprepare[daymet]"
  Free NASA Earthdata Login: https://urs.earthdata.nasa.gov/
  First-time auth will create/update ~/.netrc.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from tqdm import tqdm


# NASA CMR short name for Daymet V4 R1 Daily
DAYMET_SHORT_NAME = "Daymet_Daily_V4R1_2129"

# Output raster settings (bbox-only, 0.05 deg, EPSG:4326)
NODATA = -9999.0
CELL_SIZE = 0.05

# Fallback Daymet LCC projection (used only if rioxarray cannot auto-detect
# the CRS from the NetCDF grid_mapping variable).
DAYMET_LCC_PROJ4 = (
    "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


def _authenticate(params):
    """Log in to NASA Earthdata via earthaccess. Returns the earthaccess module."""
    try:
        import earthaccess
    except ImportError as e:
        raise ImportError(
            "Daymet requires optional deps. Install with: "
            'pip install "geoprepare[daymet]"'
        ) from e

    params.logger.info("Authenticating with NASA Earthdata Login (.netrc)...")
    auth = earthaccess.login(strategy="netrc", persist=True)
    if not auth or not getattr(auth, "authenticated", True):
        raise RuntimeError(
            "NASA Earthdata Login failed. Create an account at "
            "https://urs.earthdata.nasa.gov/ and configure ~/.netrc."
        )
    return earthaccess


def _find_opendap_urls(ea, year, bbox, params):
    """Query CMR and return OPeNDAP URLs for Daymet granules in the year/bbox."""
    results = ea.search_data(
        short_name=DAYMET_SHORT_NAME,
        temporal=(f"{year}-01-01", f"{year}-12-31"),
        bounding_box=tuple(bbox),  # (west, south, east, north)
    )
    urls = []
    for g in results:
        for link in ea.data_links(g, access="opendap"):
            urls.append(link)
    params.logger.info(f"Daymet {year}: found {len(urls)} OPeNDAP URL(s)")
    return urls


def _select_url_for_var(urls, var):
    """Pick the URL whose filename mentions this variable (Daymet files are per-var)."""
    for u in urls:
        if f"_{var}_" in os.path.basename(u):
            return u
    return None


def _build_output_profile(bbox):
    """Build a 0.05 deg EPSG:4326 GeoTIFF profile covering the bbox."""
    west, south, east, north = bbox
    width = int(np.ceil((east - west) / CELL_SIZE))
    height = int(np.ceil((north - south) / CELL_SIZE))
    transform = Affine(CELL_SIZE, 0.0, west, 0.0, -CELL_SIZE, north)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": NODATA,
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
    }
    return profile, width, height


def _pydatetime_from_time(t):
    """Convert cftime or numpy.datetime64 to a plain python datetime."""
    # cftime object path (preferred — Daymet uses noleap calendar)
    try:
        return datetime(t.year, t.month, t.day)
    except AttributeError:
        pass
    # numpy.datetime64 path
    secs = (t - np.datetime64("1970-01-01")).astype("timedelta64[s]").astype(int)
    return datetime.utcfromtimestamp(int(secs))


def _is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _process_url(url, var, year, out_dir, profile, w, h, params):
    """Open one annual Daymet OPeNDAP URL and write per-day GeoTIFFs."""
    import xarray as xr
    import rioxarray  # noqa: F401 - registers the .rio accessor on xarray
    from rasterio.enums import Resampling

    params.logger.info(f"Opening {os.path.basename(url)} via xarray+pydap...")
    ds = xr.open_dataset(url, engine="pydap", decode_times=True)
    try:
        if var not in ds:
            params.logger.error(
                f"Variable '{var}' not in dataset (have {list(ds.data_vars)})"
            )
            return
        da = ds[var]  # dims: (time, y, x) in LCC

        # Make sure rioxarray has a CRS to work with. Daymet stores it in a
        # `lambert_conformal_conic` grid_mapping variable; rioxarray normally
        # picks this up automatically, but fall back to an explicit PROJ string.
        try:
            if not da.rio.crs:
                da = da.rio.write_crs(DAYMET_LCC_PROJ4)
        except Exception:
            da = da.rio.write_crs(DAYMET_LCC_PROJ4)

        n_days = int(da.sizes["time"])
        with tqdm(total=n_days, desc=f"  Daymet {year}/{var}", leave=False) as pbar:
            for i in range(n_days):
                date = _pydatetime_from_time(da.time.values[i])
                doy = date.timetuple().tm_yday
                out_path = out_dir / f"daymet_{var}_{year}{doy:03d}_bbox.tif"
                if out_path.exists():
                    pbar.update(1)
                    continue

                slab = da.isel(time=i)
                slab_ll = slab.rio.reproject(
                    dst_crs="EPSG:4326",
                    shape=(h, w),
                    transform=profile["transform"],
                    resampling=Resampling.bilinear,
                    nodata=NODATA,
                )
                arr = np.asarray(slab_ll.values, dtype="float32")
                arr = np.where(np.isfinite(arr), arr, NODATA)

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(arr, 1)
                pbar.update(1)

        # Leap-year fix: duplicate DOY 365 -> DOY 366 (Daymet is noleap)
        if _is_leap(year):
            src = out_dir / f"daymet_{var}_{year}365_bbox.tif"
            dst = out_dir / f"daymet_{var}_{year}366_bbox.tif"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                params.logger.info(
                    f"  Duplicated DOY 365 -> 366 for leap year {year}"
                )
    finally:
        ds.close()


def run(geoprep):
    """
    Main entry point called from geodownload.py.

    Expected attributes on geoprep:
      - start_year, end_year
      - dir_intermed, logger, parser
      - daymet_bbox     : list[float]  [west, south, east, north]
      - daymet_variables: list[str]    e.g. ['tmin', 'tmax', 'prcp']
    """
    ea = _authenticate(geoprep)

    bbox = geoprep.daymet_bbox
    variables = geoprep.daymet_variables
    geoprep.logger.info(
        f"Daymet: years {geoprep.start_year}-{geoprep.end_year}, "
        f"vars {variables}, bbox {bbox}"
    )

    profile, w, h = _build_output_profile(bbox)

    for year in range(geoprep.start_year, geoprep.end_year + 1):
        try:
            urls = _find_opendap_urls(ea, year, bbox, geoprep)
        except Exception as e:
            geoprep.logger.error(f"Daymet CMR search failed for {year}: {e}")
            continue

        for var in variables:
            url = _select_url_for_var(urls, var)
            if not url:
                geoprep.logger.warning(
                    f"Daymet {year}: no URL for variable '{var}'"
                )
                continue
            out_dir = Path(geoprep.dir_intermed) / f"daymet_{var}" / str(year)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                _process_url(url, var, year, out_dir, profile, w, h, geoprep)
            except Exception as e:
                geoprep.logger.error(f"Daymet {year}/{var} failed: {e}")
                continue
