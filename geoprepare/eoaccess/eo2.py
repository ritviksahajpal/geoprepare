import os
import logging
import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
import geopandas as gpd

# For parallel downloads
from multiprocessing import Pool, cpu_count

# NASA Earthdata
import earthaccess as ea


def get_bbox_from_shapefile(shapefile_path: str, target_crs: str = "EPSG:4326"):
    """
    Computes the bounding box (minx, miny, maxx, maxy) from the given shapefile.
    Optionally reprojects the shapefile to `target_crs` first.

    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile.
    target_crs : str
        The CRS to which the shapefile will be projected before computing bounding box.
        By default, uses EPSG:4326 (latitude-longitude).

    Returns
    -------
    tuple
        A 4-element tuple (minx, miny, maxx, maxy) of the bounding box.
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Reproject if needed
    if gdf.crs and gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Compute bounding box
    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx, miny, maxx, maxy)


# -----------------------------------------------------------------------------
# 1. DEFINITIONS FOR HLS L30 vs S30
# -----------------------------------------------------------------------------
# Different band codes for L30 (Landsat) vs S30 (Sentinel-2).
# Adjust as necessary for your needs (e.g., SWIR2 = B07 or B12, etc.).

BAND_PATTERNS = {
    "L30": {
        "blue":   "B02",
        "green":  "B03",
        "red":    "B04",
        "nir":    "B05",  # Landsat’s NIR
        "swir":   "B06",  # SWIR1 in L30 (B06). If you want SWIR2, that’d be B07.
        "fmask":  "Fmask"
    },
    "S30": {
        "blue":   "B02",
        "green":  "B03",
        "red":    "B04",
        "nir":    "B08",  # Sentinel-2’s NIR
        "swir":   "B11",  # SWIR1 in S30. If you need SWIR2, that’s B12.
        "rededge":"B05",  # S2 red-edge band (optional)
        "fmask":  "Fmask"
    }
}

# Which bands each index requires (the actual formulas are the same).
# The script will map them to either L30 or S30 codes as needed.
INDEX_TO_BANDS = {
    "NDVI":   ["red", "nir"],
    "GCVI":   ["green", "nir"],
    "EVI":    ["red", "nir", "green", "blue"],
    "SAVI":   ["red", "nir"],
    "MSAVI":  ["red", "nir"],
    "NDWI":   ["green", "nir"],
    "GNDVI":  ["green", "nir"],
    "ARVI":   ["red", "nir", "blue"],
    "NDMI":   ["nir", "swir"],
    "RENDVI": ["nir", "rededge"],
    "VARI":   ["red", "nir", "green", "blue"],
}

# -----------------------------------------------------------------------------
# 2. QUALITY MASK UTILITY
# -----------------------------------------------------------------------------
def create_quality_mask(fmask_data, bit_nums=[1, 2, 3, 4, 5]):
    """
    Creates a boolean mask (True = bad pixel) from Fmask data.
    Adjust bit_nums for your product’s bit definitions if needed.
    """
    mask_array = np.zeros(fmask_data.shape, dtype=bool)
    fmask_data = np.nan_to_num(fmask_data, nan=0).astype(np.int32)

    for bit in bit_nums:
        mask_temp = (fmask_data & (1 << bit)) > 0
        mask_array = np.logical_or(mask_array, mask_temp)

    return mask_array

# -----------------------------------------------------------------------------
# 3. DETECT WHICH HLS PRODUCT (L30 or S30)
# -----------------------------------------------------------------------------
def detect_hls_product(mosaic_dir):
    """
    Scans the mosaic directory for .tif files that contain either
    'HLS.L30' or 'HLS.S30' in their filename. Returns 'L30', 'S30',
    'both', or 'unknown' depending on what it finds.

    If you have only one product type, it returns that type.
    If it finds both L30 and S30 files, returns 'both'.
    """
    mosaic_dir = Path(mosaic_dir)
    if not mosaic_dir.is_dir():
        return "unknown"

    found_l30 = False
    found_s30 = False

    for f in mosaic_dir.glob("*.tif"):
        fname = f.name
        if "HLS.L30" in fname:
            found_l30 = True
        if "HLS.S30" in fname:
            found_s30 = True

        # If we see both in at least one pass, can break
        if found_l30 and found_s30:
            return "both"

    if found_l30 and not found_s30:
        return "L30"
    elif found_s30 and not found_l30:
        return "S30"
    else:
        return "unknown"

# -----------------------------------------------------------------------------
# 4. COMPUTE INDICES FUNCTION (Works for L30 or S30)
# -----------------------------------------------------------------------------
def compute_indices_from_mosaics(
    mosaic_dir: str,
    output_dir: str,
    selected_indices: list,
    apply_fmask: bool = True,
    scale_factor: float = 0.0001
):
    """
    Looks for mosaic files in `mosaic_dir`, computes vegetation indices in
    `selected_indices`, and saves them to `output_dir`. If an Fmask mosaic
    is found (and `apply_fmask=True`), it applies QA masking.

    Detects automatically whether these mosaics are from L30 or S30 by scanning
    filenames for 'HLS.L30' or 'HLS.S30', and uses the appropriate band patterns.
    """
    mosaic_dir = Path(mosaic_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Detect the product type
    product_type = detect_hls_product(mosaic_dir)
    if product_type == "unknown":
        logging.error("Could not detect if mosaic is L30 or S30. Check file names.")
        return
    elif product_type == "both":
        logging.error("Found both L30 and S30 files in mosaic_dir. Handle them separately.")
        return

    # 2. Based on product_type, pick the correct band patterns
    patterns = BAND_PATTERNS[product_type]  # e.g., BAND_PATTERNS["L30"]

    # 3. Gather all .tif in mosaic_dir
    mosaic_files = list(mosaic_dir.glob("*.tif"))
    if not mosaic_files:
        logging.error(f"No .tif files found in {mosaic_dir}.")
        return

    # Helper to locate a mosaic file by pattern
    def find_mosaic_file(band_key):
        pattern = patterns.get(band_key, None)
        if not pattern:
            return None
        for f in mosaic_files:
            if pattern in f.name:
                return f
        return None

    # Attempt to find Fmask mosaic if apply_fmask is True
    fmask_file = None
    if apply_fmask and "fmask" in patterns:
        fmask_file = find_mosaic_file("fmask")
        if fmask_file is None:
            logging.warning("No Fmask mosaic found; proceeding without QA mask.")
    else:
        logging.info("Fmask masking disabled or no fmask pattern defined.")

    # 4. Compute each requested index
    for index_name in tqdm(selected_indices, desc=f"Computing {product_type} Indices"):
        output_file = output_dir / f"{index_name}.tif"

        # Skip if it already exists
        if output_file.exists():
            logging.info(f"{index_name} already exists at {output_file}, skipping.")
            continue

        required_bands = INDEX_TO_BANDS.get(index_name)
        if not required_bands:
            logging.error(f"Index '{index_name}' not recognized.")
            continue

        # Gather band files
        band_paths = {}
        missing_bands = []
        for b in required_bands:
            fpath = find_mosaic_file(b)
            if fpath is None:
                missing_bands.append(b)
            else:
                band_paths[b] = fpath

        if missing_bands:
            logging.error(f"Cannot compute {index_name}: missing {missing_bands}")
            continue

        # Open each band, apply scale factor
        opened_bands = {}
        for b, path in band_paths.items():
            opened_bands[b] = rxr.open_rasterio(path).squeeze() * scale_factor

        # If Fmask is found, build QA mask (True=bad)
        mask_layer = None
        if fmask_file is not None:
            fmask_data = rxr.open_rasterio(fmask_file).squeeze()
            mask_layer = create_quality_mask(fmask_data)

        # Shorthand references
        red     = opened_bands.get("red")
        nir     = opened_bands.get("nir")
        green   = opened_bands.get("green")
        blue    = opened_bands.get("blue")
        swir    = opened_bands.get("swir")
        rededge = opened_bands.get("rededge")

        # Compute the index formula
        if index_name == "NDVI":
            index_data = (nir - red) / (nir + red)
        elif index_name == "GCVI":
            index_data = (nir / green) - 1
        elif index_name == "EVI":
            index_data = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
        elif index_name == "SAVI":
            L = 0.5
            index_data = ((nir - red) / (nir + red + L)) * (1 + L)
        elif index_name == "MSAVI":
            index_data = (
                2 * nir + 1
                - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))
            ) / 2
        elif index_name == "NDWI":
            index_data = (green - nir) / (green + nir)
        elif index_name == "GNDVI":
            index_data = (nir - green) / (nir + green)
        elif index_name == "ARVI":
            index_data = (nir - (2*red - blue)) / (nir + (2*red - blue))
        elif index_name == "NDMI":
            index_data = (nir - swir) / (nir + swir)
        elif index_name == "RENDVI":
            index_data = (nir - rededge) / (nir + rededge)
        elif index_name == "VARI":
            index_data = (green - red) / (green + red - blue)
        else:
            logging.error(f"Index '{index_name}' is not implemented.")
            continue

        # Apply QA mask if present
        if mask_layer is not None:
            index_data = index_data.where(~mask_layer)

        # Save final index raster
        index_data.rio.to_raster(output_file)
        logging.info(f"{index_name} saved => {output_file}")

# -----------------------------------------------------------------------------
# 5. NASA EARTHACCESS CLASS (Download)
# -----------------------------------------------------------------------------
@dataclass
class NASAEarthAccess:
    dataset: list            # e.g. ["HLSL30"] or ["HLSS30"] or both
    bbox: tuple = None       # (minx, miny, maxx, maxy)
    shapefile: Path = None   # If deriving bbox from shapefile
    temporal: tuple = None   # (start_date, end_date)
    output_dir: Path = None  # Download folder
    log_dir: str = None
    logging_level: int = logging.WARNING
    logging_project: str = "eo_access"
    logging_file: str = os.path.splitext(os.path.basename(__file__))[0]
    login_strategy: str = "netrc"

    def __post_init__(self):
        self.results: list = None
        if self.bbox and self.shapefile:
            raise ValueError("Cannot specify both bbox and shapefile.")

        if self.shapefile:
            self._get_bbox_from_shapefile()

        # Earthaccess login
        auth = ea.login(strategy=self.login_strategy)
        if not auth.authenticated:
            ea.login(strategy="interactive", persist=True)

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _get_bbox_from_shapefile(self):
        try:
            gdf = gpd.read_file(self.shapefile)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            self.bbox = tuple(gdf.total_bounds)
        except Exception as e:
            raise RuntimeError(f"Error reading shapefile: {e}")

    def search_data(self):
        try:
            self.results = ea.search_data(
                short_name=self.dataset,
                bounding_box=self.bbox,
                temporal=self.temporal
            )
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    @staticmethod
    def _download_item(item):
        res, outdir = item
        ea.download([res], str(outdir))

    def download_parallel(self):
        if not self.results:
            raise RuntimeError("No results found. Did you run `search_data()`?")

        combos = [(r, self.output_dir) for r in self.results]
        num_cpu = max(1, int(cpu_count() * 0.6))

        with Pool(num_cpu) as pool:
            for _ in tqdm(
                pool.imap_unordered(self._download_item, combos),
                total=len(combos),
                desc="Downloading HLS"
            ):
                pass

    def download_single(self):
        """
        Download one item at a time (serially) and use tqdm to show progress.
        This method is called by default.
        """
        if not self.results:
            raise RuntimeError("No results found. Did you run search_data()?")

        for result in tqdm(self.results, desc="Downloading items"):
            try:
                ea.download([result], str(self.output_dir))
            except Exception as e:
                logging.error(f"Download failed: {e}")
                continue

# -----------------------------------------------------------------------------
# 6. EARTHACCESS PROCESSOR (Mosaic)
# -----------------------------------------------------------------------------
@dataclass
class EarthAccessProcessor:
    def __init__(self, input_dir: str, mosaic_dir: str):
        """
        Parameters:
        -----------
        input_dir : str
            Directory where downloaded .tif files are stored.
        mosaic_dir : str
            Directory where the output mosaics will be saved.
        """
        self.input_dir = Path(input_dir)
        self.mosaic_dir = Path(mosaic_dir)
        self.mosaic_dir.mkdir(exist_ok=True, parents=True)

    def group_files_by_band_and_date(self):
        # (Same as before) ...
        grouped_files = defaultdict(list)
        for filepath in self.input_dir.glob("*.tif"):
            fname = filepath.name
            parts = fname.split(".")
            if len(parts) < 7:
                continue
            band = parts[-2]
            date_token = parts[3]
            date = date_token[:8]
            grouped_files[(band, date)].append(filepath)
        return grouped_files

    def mosaic(self):
        """
        For each (band, date) group, merges the .tif files into one output.
        If some files have a different CRS, they will be reprojected to match
        the first dataset's CRS and resolution.
        """
        from rioxarray.merge import merge_arrays
        from rasterio.enums import Resampling  # for choosing the resampling method

        grouped_files = self.group_files_by_band_and_date()

        for (band, date), files in tqdm(grouped_files.items(), desc="Mosaicing by band and date"):
            # Determine product tag from the first file's name
            product_tag = ""
            if files:
                first_fname = files[0].name
                if "HLS.L30" in first_fname:
                    product_tag = "HLS.L30_"
                elif "HLS.S30" in first_fname:
                    product_tag = "HLS.S30_"

            output_file = self.mosaic_dir / f"mosaic_{product_tag}{band}_{date}.tif"
            if output_file.exists():
                logging.info(f"Mosaic {output_file} already exists, skipping.")
                continue

            ds_list = [rxr.open_rasterio(fp) for fp in files]
            if not ds_list:
                logging.warning(f"No valid datasets for band={band}, date={date}. Skipping.")
                continue

            # The "reference" CRS and resolution are those of the first dataset
            reference_crs = ds_list[0].rio.crs
            reference_res = ds_list[0].rio.resolution()

            # Reproject any dataset that does not match the reference
            for i in range(len(ds_list)):
                ds_crs = ds_list[i].rio.crs
                ds_res = ds_list[i].rio.resolution()

                if ds_crs != reference_crs or ds_res != reference_res:
                    logging.info(
                        f"Reprojecting {files[i].name} from {ds_crs} to {reference_crs} "
                        f"and from res={ds_res} to {reference_res}."
                    )
                    ds_list[i] = ds_list[i].rio.reproject(
                        dst_crs=reference_crs,
                        resolution=reference_res,
                        resampling=Resampling.nearest  # choose your resampling method
                    )

            # Now all datasets share the same CRS/res, we can mosaic safely
            mosaic_data = merge_arrays(ds_list)
            mosaic_data.rio.to_raster(output_file)
            logging.info(f"Saved mosaic => {output_file}")


def run(shapefile_path: Path,
        download_dir: Path,
        mosaic_dir: Path,
        indices_dir: Path,
        indices: list,
        datasets: list,
        temporal: tuple):
    bbox = get_bbox_from_shapefile(shapefile_path)
    print(bbox)  # (minx, miny, maxx, maxy) in EPSG:4326

    # 3. Create the NASA EarthAccess object
    #    You can specify L30 or S30 or BOTH:
    #    e.g. dataset=["HLSL30","HLSS30"] to get both.
    hls = NASAEarthAccess(
        dataset=datasets,  # or ["HLSS30"] or both
        bbox=bbox,
        temporal=temporal,  # example range
        output_dir=download_dir
    )

    # 4. Search & Download
    hls.search_data()
    hls.download_single()

    # 5. Mosaic
    processor = EarthAccessProcessor(
        input_dir=download_dir,
        mosaic_dir=mosaic_dir
    )
    processor.mosaic()

    # 6. Compute Indices
    #    The function will automatically detect L30 vs S30 in mosaic_dir
    #    and use the correct band mappings for your chosen indices.
    compute_indices_from_mosaics(
        mosaic_dir=mosaic_dir,
        output_dir=indices_dir,
        selected_indices=indices,
        apply_fmask=True,  # if a mosaic_Fmask.tif is found
        scale_factor=0.0001  # typical for HLS reflectance
    )

    print("Done!")
    print(f"Downloaded data  => {download_dir}")
    print(f"Mosaic outputs   => {mosaic_dir}")
    print(f"Computed indices => {indices_dir}")

# -----------------------------------------------------------------------------
# 7. EXAMPLE MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(r"D:\Users\ritvik\projects\GEOGLAM\Input\Global_Datasets\Regions\Shps")
    run(shapefile_path=base_dir / "wolayita_dissolved.shp",
        download_dir=base_dir / "hls_download",
        mosaic_dir=base_dir / "hls_mosaics",
        indices_dir=base_dir / "hls_indices",
        indices=["NDVI", "EVI", "SAVI", "NDMI"],
        datasets=["HLSL30", "HLSS30"],
        temporal=("2020-01-01", "2025-01-30"))
