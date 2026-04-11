#!/usr/bin/env python3
"""
Ritvik Sahajpal
ritvik@umd.edu

Download AlphaEarth Foundations (AEF) satellite embedding TIF files.

Data source: https://source.coop/tge-labs/aef

The AEF dataset contains annual satellite embeddings from 2018-2024 (64 channels per pixel).
Files are organized by year and UTM zone.

The COGs in this dataset are "bottom-up" where the origin is the bottom-left corner.
VRT files are provided to correct this on-the-fly - always use VRTs for data access.

Steps:
 1. Load the AEF tile index (CSV format with WGS84 geometries)
 2. Fetch manifest to get actual file paths
 3. Find tiles that intersect with specified country/region extent
 4. Download matching .tiff and .vrt files

License: CC-BY 4.0
Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by 
             Google and Google DeepMind."
"""
import logging
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import requests
from shapely.geometry import box
from tqdm import tqdm

# Module-level logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# Base URL for AEF data
AEF_BASE_URL = "https://data.source.coop/tge-labs/aef/v1/annual"
AEF_NUM_BANDS = 64
AEF_INDEX_PARQUET_URL = f"{AEF_BASE_URL}/aef_index.parquet"


def get_country_lat_lon_extent(
    country_names: Union[str, List[str]], buffer: float = 0.5
) -> List[float]:
    """
    Get the bounding box for one or more countries.

    Args:
        country_names: Country name(s) (e.g., "Kenya" or ["Kenya", "Tanzania"])
        buffer: Buffer in degrees to add around the bounding box

    Returns:
        [west, east, south, north] in WGS84
    """
    if isinstance(country_names, str):
        country_names = [country_names]

    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url, engine="pyogrio")

    minx, maxx, miny, maxy = 180, -180, 90, -90
    country_found = False

    for country_name in country_names:
        country_name_normalized = country_name.lower().replace(" ", "_")

        # Handle special cases
        if country_name_normalized in ["russia", "russian_federation"]:
            return [20, 80, 40, 80]

        # Map alternative names
        name_mapping = {
            "dem_people's_rep_of_korea": "north_korea",
            "republic_of_korea": "south_korea",
            "united_republic_of_tanzania": "tanzania",
            "ivory_coast": "côte_d'ivoire",
            "cote_d'ivoire": "côte_d'ivoire",
        }
        country_name_normalized = name_mapping.get(
            country_name_normalized, country_name_normalized
        )

        country = world[
            world.ADMIN.str.lower().str.replace(" ", "_") == country_name_normalized
        ]

        if not country.empty:
            country_found = True
            bbox = country.bounds.iloc[0]
            minx = min(minx, bbox.minx)
            maxx = max(maxx, bbox.maxx)
            miny = min(miny, bbox.miny)
            maxy = max(maxy, bbox.maxy)

    if country_found:
        return [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]
    else:
        logger.warning(f"Country '{country_names}' not found. Returning global extent.")
        return [-180, 180, -90, 90]



def load_aef_index(cache_path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """
    Load the AEF index file containing tile geometries and metadata.

    The GeoParquet index contains one entry per file with:
    - geometry: WGS84 polygon of tile coverage
    - path: S3 URL to the .tiff file
    - location: VRT path for the file (corrects bottom-up COG orientation)
    - crs, year, utm_zone, utm/wgs84 bounds

    Args:
        cache_path: Optional local path to cache/load the index

    Returns:
        GeoDataFrame with tile information
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info(f"Loading cached index from {cache_path}")
            gdf = gpd.read_file(cache_path) if str(cache_path).endswith(".gpkg") \
                else gpd.read_parquet(cache_path)
            # Check that the cache has valid paths (not from a failed earlier run)
            if "rel_path" in gdf.columns and gdf["rel_path"].notna().any():
                return gdf
            logger.warning("Cached index has no valid paths, re-downloading...")

    # Download the GeoParquet index (contains path column with S3 URLs)
    logger.info(f"Downloading AEF index from {AEF_INDEX_PARQUET_URL}...")
    import tempfile
    response = requests.get(AEF_INDEX_PARQUET_URL, timeout=120)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    gdf = gpd.read_parquet(tmp_path)
    os.unlink(tmp_path)

    logger.info(f"Loaded {len(gdf)} index entries")

    # Extract relative path from S3 URL in the 'path' column
    # e.g. "s3://...v1/annual/2018/1N/file.tiff" -> "2018/1N/file.tiff"
    if "path" in gdf.columns:
        gdf["rel_path"] = gdf["path"].apply(
            lambda p: p.split("v1/annual/", 1)[1] if "v1/annual/" in str(p) else str(p)
        )
        matched = gdf["rel_path"].notna().sum()
        logger.info(f"Extracted relative paths for {matched}/{len(gdf)} entries")
    else:
        logger.error("Index parquet has no 'path' column. Available: " +
                      ", ".join(gdf.columns))
        gdf["rel_path"] = None

    # Cache if requested
    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(gdf)} rows to cache {cache_path} ...")
        if str(cache_path).endswith(".parquet"):
            gdf.to_parquet(cache_path)
        else:
            gdf.to_file(cache_path, driver="GPKG")
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Cached index to {cache_path} ({size_mb:.1f} MB)")

    return gdf


def find_tiles_for_extent(
    index_gdf: gpd.GeoDataFrame,
    extent: List[float],
    years: Optional[List[int]] = None,
) -> gpd.GeoDataFrame:
    """
    Find all tiles that intersect with a given extent.

    Args:
        index_gdf: GeoDataFrame with tile index
        extent: [west, east, south, north] in WGS84
        years: Optional list of years to filter

    Returns:
        GeoDataFrame of matching tiles
    """
    west, east, south, north = extent
    bbox_geom = box(west, south, east, north)

    # Filter by spatial intersection
    matching_tiles = index_gdf[index_gdf.intersects(bbox_geom)].copy()

    # Filter by years if specified
    if years is not None:
        matching_tiles = matching_tiles[matching_tiles["year"].isin(years)]

    # Filter to tiles that have valid paths
    if "rel_path" in matching_tiles.columns:
        valid_paths = matching_tiles["rel_path"].notna()
        if not valid_paths.all():
            n_invalid = (~valid_paths).sum()
            logger.warning(f"{n_invalid} tiles have no matching file path")
        matching_tiles = matching_tiles[valid_paths]

    return matching_tiles


def download_file(
    url: str, output_path: Path, overwrite: bool = False
) -> Tuple[bool, str]:
    """
    Download a file from URL to local path.

    Returns:
        Tuple of (success, error_message or empty string)
    """
    if output_path.exists() and not overwrite:
        return True, ""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, ""
    except Exception as e:
        return False, str(e)


def _download_file_wrapper(args):
    """Wrapper for multiprocessing."""
    url, output_path, overwrite = args
    return download_file(url, output_path, overwrite)


def download_aef_tiles(
    matching_tiles: gpd.GeoDataFrame,
    output_dir: Path,
    country: str = "",
    download_vrt: bool = True,
    overwrite: bool = False,
    parallel_process: bool = False,
    fraction_cpus: float = 0.75,
) -> Tuple[List[Path], List[Tuple[str, str]]]:
    """
    Download AEF tiles.

    Args:
        matching_tiles: GeoDataFrame of tiles to download
        output_dir: Directory to save downloaded files
        country: Country name (for log messages)
        download_vrt: Whether to also download VRT files
        overwrite: Whether to overwrite existing files
        parallel_process: Whether to use parallel processing
        fraction_cpus: Fraction of CPUs to use

    Returns:
        Tuple of (downloaded_files, failed_downloads)
    """
    # Prepare download list
    downloads = []
    for _, row in matching_tiles.iterrows():
        rel_path = row["rel_path"]
        tif_url = f"{AEF_BASE_URL}/{rel_path}"
        local_tif_path = output_dir / rel_path
        downloads.append((tif_url, local_tif_path, overwrite))

        if download_vrt:
            vrt_path = rel_path.replace(".tiff", ".vrt")
            vrt_url = f"{AEF_BASE_URL}/{vrt_path}"
            local_vrt_path = output_dir / vrt_path
            downloads.append((vrt_url, local_vrt_path, overwrite))

    country_tag = f" for {country}" if country else ""
    logger.info(f"Downloading {len(downloads)} files{country_tag}...")

    downloaded_files = []
    failed_downloads = []

    if parallel_process:
        num_workers = max(1, int(multiprocessing.cpu_count() * fraction_cpus))
        with multiprocessing.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_download_file_wrapper, downloads),
                    total=len(downloads),
                    desc="Downloading AEF tiles",
                )
            )
            for (url, path, _), (success, error) in zip(downloads, results):
                if success:
                    downloaded_files.append(path)
                else:
                    failed_downloads.append((url, error))
    else:
        for url, path, ow in tqdm(downloads, desc="Downloading AEF tiles"):
            success, error = download_file(url, path, ow)
            if success:
                downloaded_files.append(path)
            else:
                failed_downloads.append((url, error))

    return downloaded_files, failed_downloads


def to_global(params, year, country):
    """
    Resample downloaded AEF tiles for a given year to the standard
    0.05-degree global grid (3600 x 7200 pixels, EPSG:4326).

    Each 0.05-degree cell is the average of all underlying ~10 m pixels.
    Areas with no data are set to NaN.

    Output: dir_intermed/aef/{country}/aef_{year}_{country}.tif  (64-band Float32 GeoTIFF)
    """
    from osgeo import gdal

    # Prevent GDAL from attempting S3 access (e.g. sidecar files, overviews)
    gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")

    country_slug = country.lower().replace(" ", "_")
    aef_download_dir = Path(params.dir_download) / "aef"
    output_dir = Path(params.dir_intermed) / "aef" / country_slug / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"aef_{year}_{country_slug}.tif"
    redo = getattr(params, "redo", False)
    if output_file.exists() and not redo:
        params.logger.info(f"Skipping {output_file} (already exists)")
        return

    # Use locally downloaded TIFF files directly. gdal.Warp uses the
    # geotransform for correct spatial placement regardless of pixel order
    # (bottom-up COGs are handled correctly).
    year_dir = aef_download_dir / str(year)
    if not year_dir.exists():
        params.logger.warning(f"No download directory for year {year}")
        return

    tiff_files = sorted(year_dir.rglob("*.tiff"))
    if not tiff_files:
        params.logger.warning(f"No TIFF files found for year {year}")
        return
    input_paths = [str(f) for f in tiff_files]
    params.logger.info(f"Found {len(input_paths)} TIFF files for {year}")

    # Warp all tiles to the standard 0.05° global grid with average resampling
    warp_options = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        xRes=0.05,
        yRes=0.05,
        outputBounds=(-180, -90, 180, 90),  # full global extent
        resampleAlg="average",
        outputType=gdal.GDT_Float32,
        dstNodata=float("nan"),
        creationOptions=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "TILED=YES",
            "BIGTIFF=YES",
        ],
        multithread=True,
    )

    params.logger.info(
        f"Warping {len(input_paths)} tiles to global 0.05° grid -> {output_file}"
    )
    ds = gdal.Warp(str(output_file), input_paths, options=warp_options)
    if ds is None:
        params.logger.error(f"gdal.Warp failed for year {year}")
        return
    ds = None  # flush and close

    params.logger.info(f"Created {output_file}")


def _country_output_dir(dir_intermed, country):
    """Return the per-country AEF output directory."""
    country_slug = country.lower().replace(" ", "_")
    return Path(dir_intermed) / "aef" / country_slug


def _all_yearly_files_exist(dir_intermed, country, years):
    """Check if all per-country yearly AEF files already exist."""
    out_dir = _country_output_dir(dir_intermed, country)
    country_slug = country.lower().replace(" ", "_")
    return all((out_dir / str(y) / f"aef_{y}_{country_slug}.tif").exists() for y in years)


def compute_average_aef(params, country, years):
    """
    Average yearly AEF TIFs into a single multi-year mean file.

    Reads aef_{year}_{country}.tif for each year, computes per-band pixel-wise
    nanmean across years, and writes aef_avg_{country}.tif in the same directory.
    """
    out_dir = _country_output_dir(params.dir_intermed, country)
    country_slug = country.lower().replace(" ", "_")
    avg_file = out_dir / f"aef_avg_{country_slug}.tif"

    redo = getattr(params, "redo", False)
    if avg_file.exists() and not redo:
        params.logger.info(f"Skipping {avg_file} (already exists)")
        return

    # Collect existing yearly files
    yearly_files = []
    for y in years:
        f = out_dir / str(y) / f"aef_{y}_{country_slug}.tif"
        if f.exists():
            yearly_files.append(f)

    if not yearly_files:
        params.logger.warning(f"No yearly AEF files found for {country}, skipping average")
        return

    params.logger.info(
        f"Computing average AEF from {len(yearly_files)} files for {country}..."
    )

    # Read first file to get metadata (shape, transform, crs, nodata)
    with rasterio.open(yearly_files[0]) as src:
        meta = src.meta.copy()
        n_bands = src.count
        height = src.height
        width = src.width

    # Accumulate sum and count per pixel per band (avoids loading all years at once)
    band_sum = np.zeros((n_bands, height, width), dtype=np.float64)
    band_count = np.zeros((n_bands, height, width), dtype=np.int32)

    for f in tqdm(yearly_files, desc=f"Reading AEF years ({country})"):
        with rasterio.open(f) as src:
            for b in range(1, n_bands + 1):
                data = src.read(b).astype(np.float64)
                valid = np.isfinite(data)
                band_sum[b - 1] += np.where(valid, data, 0.0)
                band_count[b - 1] += valid.astype(np.int32)

    # Compute mean, set pixels with no valid observations to NaN
    with np.errstate(invalid="ignore"):
        band_mean = np.where(band_count > 0, band_sum / band_count, np.nan)

    # Write output
    meta.update(dtype="float32", count=n_bands, nodata=float("nan"))
    with rasterio.open(avg_file, "w", **meta) as dst:
        for b in range(n_bands):
            dst.write(band_mean[b].astype(np.float32), b + 1)

    params.logger.info(f"Created {avg_file}")


def _init_gee():
    """Initialize Google Earth Engine. Returns True if successful."""
    try:
        import ee
        ee.Initialize()
        return True
    except ImportError:
        logger.warning(
            "earthengine-api not installed. Install with: pip install earthengine-api"
        )
        return False
    except Exception as e:
        logger.warning(f"GEE initialization failed: {e}. Run ee.Authenticate() first.")
        return False


def download_aef_gee(params, country, year, output_file, extent):
    """
    Download AEF embeddings for a country/year via Google Earth Engine.

    Server-side: mosaic + reduceResolution(mean) to 0.05 deg.
    Downloads as numpy array, writes GeoTIFF with rasterio.
    """
    import ee
    import io
    import math
    from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, max=20),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _compute_pixels(request):
        return ee.data.computePixels(request)

    west, east, south, north = extent
    res = 0.05

    # Snap extent to 0.05 deg grid
    west_s = math.floor(west / res) * res
    south_s = math.floor(south / res) * res
    east_s = math.ceil(east / res) * res
    north_s = math.ceil(north / res) * res

    width = round((east_s - west_s) / res)
    height = round((north_s - south_s) / res)

    params.logger.info(
        f"GEE grid: {width}x{height} pixels, "
        f"extent [{west_s:.2f}, {south_s:.2f}, {east_s:.2f}, {north_s:.2f}]"
    )

    # Build GEE image: mosaic -> reduceResolution -> reproject
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    image = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
    image = image.reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=True,
        maxPixels=65536,
    ).reproject(crs="EPSG:4326", crsTransform=[res, 0, -180, 0, -res, 90])

    band_names = [f"A{i:02d}" for i in range(AEF_NUM_BANDS)]

    # Download in latitude strips if the response would exceed ~40 MB
    MAX_BYTES = 40 * 1024 * 1024
    bytes_per_row = width * AEF_NUM_BANDS * 4  # float32
    max_rows = max(1, MAX_BYTES // bytes_per_row)

    result = np.full((AEF_NUM_BANDS, height, width), np.nan, dtype=np.float32)

    row_offset = 0
    strip_north = north_s
    remaining = height
    strip_num = 0

    while remaining > 0:
        strip_height = min(max_rows, remaining)
        strip_num += 1

        if height > max_rows:
            params.logger.info(f"  Downloading strip {strip_num} ({strip_height} rows)...")

        request = {
            "expression": image,
            "fileFormat": "NUMPY_NDARRAY",
            "bandIds": band_names,
            "grid": {
                "dimensions": {"width": width, "height": strip_height},
                "affineTransform": {
                    "scaleX": res,
                    "shearX": 0,
                    "translateX": west_s,
                    "shearY": 0,
                    "scaleY": -res,
                    "translateY": strip_north,
                },
                "crsCode": "EPSG:4326",
            },
        }
        data = np.load(io.BytesIO(_compute_pixels(request)))

        for i, band in enumerate(band_names):
            result[i, row_offset : row_offset + strip_height, :] = data[band]

        row_offset += strip_height
        strip_north -= strip_height * res
        remaining -= strip_height

    # Write GeoTIFF
    output_file.parent.mkdir(parents=True, exist_ok=True)
    from rasterio.transform import from_origin

    transform = from_origin(west_s, north_s, res, res)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": AEF_NUM_BANDS,
        "height": height,
        "width": width,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": float("nan"),
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(result)

    params.logger.info(f"Wrote {output_file} ({AEF_NUM_BANDS} bands, {width}x{height})")


def run(geoprep):
    """
    Main entry point for AEF processing.

    Args:
        geoprep: GeoDownload object containing configuration parameters
            Required attributes:
            - start_year: First year to process
            - end_year: Last year to process
            - dir_download: Base download directory
            - parallel_process: Whether to use multiprocessing
            - fraction_cpus: Fraction of CPUs to use for parallel processing
            - aef_countries: List of country names to download (from config)
            - aef_buffer: Buffer in degrees around country extent
            - aef_download_vrt: Whether to download VRT files
            - aef_source: "gee" (Google Earth Engine) or "tiles" (source.coop HTTP)
    """
    # Extract parameters from geoprep object
    start_year = geoprep.start_year
    end_year = geoprep.end_year
    dir_download = Path(geoprep.dir_download)
    parallel_process = geoprep.parallel_process
    fraction_cpus = geoprep.fraction_cpus

    # AEF-specific parameters
    countries = getattr(geoprep, "aef_countries", [])
    buffer = getattr(geoprep, "aef_buffer", 0.5)
    download_vrt = getattr(geoprep, "aef_download_vrt", True)
    index_cache_path = getattr(geoprep, "aef_index_cache", None)

    # Validate
    if not countries:
        logger.error("No countries specified for AEF download. Set aef_countries in config.")
        return

    # Available years for AEF
    available_years = list(range(2018, 2025))  # 2018-2024, note: 2017 not available
    years_to_download = [y for y in range(start_year, end_year + 1) if y in available_years]

    if not years_to_download:
        logger.warning(f"No valid years in range {start_year}-{end_year}. AEF has 2018-2024.")
        return

    logger.info(f"Processing AEF data for countries: {countries}")
    logger.info(f"Years: {years_to_download}")

    # Check which countries actually need downloading/resampling
    redo = getattr(geoprep, "redo", False)
    countries_to_process = [
        c for c in countries
        if redo or not _all_yearly_files_exist(geoprep.dir_intermed, c, years_to_download)
    ]

    if not countries_to_process:
        logger.info("All per-country AEF files already exist, skipping download/resample")
        # Still compute averages for countries that need them
        for country in countries:
            compute_average_aef(geoprep, country, years_to_download)
        return

    logger.info(f"Countries needing download/resample: {countries_to_process}")

    # Determine data source
    aef_source = getattr(geoprep, "aef_source", "gee")
    use_gee = aef_source == "gee"

    # If GEE requested, try to initialize; fall back to tiles on failure
    if use_gee:
        if _init_gee():
            logger.info("Using Google Earth Engine for AEF download")
        else:
            logger.warning("Falling back to tile-based download")
            use_gee = False

    if use_gee:
        _run_gee(geoprep, countries, countries_to_process, years_to_download, buffer)
    else:
        _run_tiles(
            geoprep, countries, countries_to_process, years_to_download,
            buffer, download_vrt, index_cache_path, dir_download,
            parallel_process, fraction_cpus,
        )


def _run_gee(geoprep, countries, countries_to_process, years, buffer):
    """Download AEF via Google Earth Engine (server-side resample to 0.05 deg)."""
    redo = getattr(geoprep, "redo", False)

    for country in countries:
        logger.info(f"--- Processing AEF for {country} (GEE) ---")

        if country not in countries_to_process:
            logger.info(f"All yearly files exist for {country}, skipping")
            compute_average_aef(geoprep, country, years)
            continue

        extent = get_country_lat_lon_extent([country], buffer=buffer)
        logger.info(
            f"Extent (with {buffer} deg buffer): "
            f"W={extent[0]:.2f}, E={extent[1]:.2f}, "
            f"S={extent[2]:.2f}, N={extent[3]:.2f}"
        )

        country_slug = country.lower().replace(" ", "_")
        for year in tqdm(years, desc=f"GEE AEF ({country})"):
            out_dir = Path(geoprep.dir_intermed) / "aef" / country_slug / str(year)
            output_file = out_dir / f"aef_{year}_{country_slug}.tif"
            if output_file.exists() and not redo:
                logger.info(f"Skipping {output_file} (exists)")
                continue
            download_aef_gee(geoprep, country, year, output_file, extent)

        compute_average_aef(geoprep, country, years)


def _run_tiles(
    geoprep, countries, countries_to_process, years,
    buffer, download_vrt, index_cache_path, dir_download,
    parallel_process, fraction_cpus,
):
    """Download AEF tiles from source.coop and resample locally."""
    # Set up directories
    download_dir = dir_download / "aef"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Set up index cache path
    if index_cache_path:
        cache_path = Path(index_cache_path)
    else:
        cache_path = dir_download / "aef" / "aef_index_cache.gpkg"

    # Load index once (shared across countries that need processing)
    logger.info("Loading AEF tile index...")
    index_gdf = load_aef_index(cache_path=cache_path)
    logger.info(f"Total tiles in index: {len(index_gdf)}")
    logger.info(f"Years available in index: {sorted(index_gdf['year'].unique())}")

    # Process each country
    for country in countries:
        logger.info(f"--- Processing AEF for {country} (tiles) ---")

        # Skip download/resample if all yearly files exist
        if country not in countries_to_process:
            logger.info(f"All yearly files exist for {country}, skipping download/resample")
            compute_average_aef(geoprep, country, years)
            continue

        # Get country extent
        extent = get_country_lat_lon_extent([country], buffer=buffer)
        logger.info(
            f"Extent (with {buffer} deg buffer): "
            f"W={extent[0]:.4f}, E={extent[1]:.4f}, "
            f"S={extent[2]:.4f}, N={extent[3]:.4f}"
        )

        # Find matching tiles
        logger.info("Finding tiles covering the specified area...")
        matching_tiles = find_tiles_for_extent(index_gdf, extent, years=years)
        logger.info(f"Found {len(matching_tiles)} matching tiles")

        if len(matching_tiles) == 0:
            logger.warning(f"No tiles found for {country}.")
            continue

        # Summary by year
        year_counts = matching_tiles["year"].value_counts().sort_index()
        logger.info("Tiles per year:")
        for year, count in year_counts.items():
            logger.info(f"  {year}: {count} tiles")

        # Download tiles (shared dir — skips already downloaded files)
        downloaded_files, failed_downloads = download_aef_tiles(
            matching_tiles,
            download_dir,
            country=country,
            download_vrt=download_vrt,
            overwrite=False,
            parallel_process=parallel_process,
            fraction_cpus=fraction_cpus,
        )

        # Summary
        logger.info(f"Download complete for {country}!")
        logger.info(f"  Successfully downloaded: {len(downloaded_files)} files")
        logger.info(f"  Failed: {len(failed_downloads)} files")
        logger.info(f"  Output directory: {download_dir.absolute()}")

        if failed_downloads:
            logger.warning("Failed downloads:")
            for url, error in failed_downloads[:5]:
                logger.warning(f"  {url}: {error}")
            if len(failed_downloads) > 5:
                logger.warning(f"  ... and {len(failed_downloads) - 5} more")

        # Resample to 0.05 deg global grid (per country subfolder)
        logger.info(f"Resampling tiles to 0.05 deg global grid for {country}...")
        for year in tqdm(years, desc=f"Resampling AEF ({country})"):
            to_global(geoprep, year, country)

        # Compute average AEF
        compute_average_aef(geoprep, country, years)


if __name__ == "__main__":
    pass