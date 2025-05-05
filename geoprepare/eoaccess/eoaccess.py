import os
import logging
import rasterio as rio
import xarray as xr
import numpy as np
import rioxarray as rxr
from pathlib import Path
import earthaccess as ea
import geopandas as gpd
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from .. import log
from .. import utils


@dataclass
class NASAEarthAccess:
    dataset: list
    bbox: tuple = None
    shapefile: Path = None
    temporal: tuple = None
    output_dir: Path = None
    log_dir: str = None
    logging_level: int = logging.WARNING
    logging_project: str = "eo_access"
    logging_file: str = os.path.splitext(os.path.basename(__file__))[0]
    login_strategy: str = "netrc"

    def __post_init__(self):
        self.results: list = None

        # Both bbox and shapefile cannot be not None at the same time
        if self.bbox and self.shapefile:
            raise ValueError("Both bbox and shapefile cannot be specified")

        if self.shapefile:
            self.get_bbox_from_shapefile()

        auth = ea.login(strategy=self.login_strategy)
        if not auth.authenticated:
            ea.login(strategy="interactive", persist=True)

        # Change output_dir to Path if it is a string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = log.Logger(
            dir_log=self.output_dir / "logs",
            project=self.logging_project,
            file=self.logging_file,
            level=self.logging_level,
        )

    def get_bbox_from_shapefile(self):
        try:
            gdf = gpd.read_file(self.shapefile)
        except:
            raise RuntimeError(f"{self.shapefile} could not be read")

        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        self.bbox = tuple(list(gdf.total_bounds))

    def search_data(self):
        try:
            self.results = ea.search_data(
                short_name=self.dataset,
                bounding_box=self.bbox,
                temporal=self.temporal,
            )
        except Exception as e:
            raise RuntimeError(f"Search failed for {self.dataset}, error: {e}")

    def stream(self):
        fileset = ea.open(self.results)

        print(f"Using {type(fileset[0])} filesystem")
        datasets = [rxr.open_rasterio(file) for file in fileset]

        with tqdm(total=len(datasets), desc="Concatenating datasets") as pbar:

            def concat_with_progress(ds_list):
                for ds in ds_list:
                    yield ds
                    pbar.update(1)

            merged_dataset = xr.concat(
                concat_with_progress(datasets), dim="new_dimension"
            )

        ds = xr.open_mfdataset(fileset)
        return ds

    @staticmethod
    def download(item):
        result, output_dir = item
        ea.download([result], str(output_dir))

    def download_parallel(self):
        combinations = [(result, self.output_dir) for result in self.results]
        num_cpu = int(cpu_count() * 0.6)

        try:
            with Pool(num_cpu) as p:
                for _ in tqdm(
                    p.imap_unordered(self.download, combinations),
                    total=len(combinations),
                ):
                    pass
        finally:
            p.close()
            p.join()


@dataclass
class EarthAccessProcessor:
    dataset: list
    bbox: tuple = None
    shapefile: Path = None
    start_date_col: str = None
    end_date_col: str = None
    temporal: tuple = None
    input_dir: str = None

    def __post_init__(self):
        if self.bbox and self.shapefile:
            raise ValueError("Both bbox and shapefile cannot be specified")

        self.mosaic_dir = Path(self.input_dir) / "mosaic"
        os.makedirs(self.mosaic_dir, exist_ok=True)

    def get_ts(self):
        try:
            dg = gpd.read_file(self.shapefile)
            for index, row in tqdm(dg.iterrows(), desc="Getting time-series"):
                pass
        except Exception as e:
            logging.error(f"Error while processing time-series: {e}")
            raise

    def group_files_by_band_and_date(self):
        grouped_files = defaultdict(list)

        for filename in os.listdir(self.input_dir):
            if filename.endswith(".tif"):
                parts = filename.split(".")
                band = parts[-2]
                date = parts[3][:8]

                grouped_files[(band, date)].append(filename)

        return grouped_files

    def mosaic(self):
        grouped_files = self.group_files_by_band_and_date()

        for key, files in tqdm(grouped_files.items(), desc="Mosaicing files"):
            band, date = key
            mosaic_file = self.mosaic_dir / f"mosaic_{band}_{date}.tif"

            if os.path.exists(mosaic_file):
                continue

            tif_files = [Path(self.input_dir) / file for file in files]

            # Use rioxarray to mosaic the files
            first_file = rxr.open_rasterio(tif_files[0])
            crs = first_file.rio.crs
            res = first_file.rio.resolution()

            for file in tif_files[1:]:
                ds = rxr.open_rasterio(file)
                if ds.rio.crs != crs or ds.rio.resolution() != res:
                    raise ValueError(f"File {file} has different CRS or resolution")

            # Call mosaic utility function
            utils.mosaic(tif_files, mosaic_file)

    def create_quality_mask(self, quality_data, bit_nums: list = [1, 2, 3, 4, 5]):
        """
        Uses the Fmask layer and bit numbers to create a binary mask of good pixels.
        By default, bits 1-5 are used to remove bad pixels like cloud, shadow, snow.

        Parameters:
        - quality_data: The Fmask layer data (2D array).
        - bit_nums: List of bit numbers to use for masking (default: bits 1-5).

        Returns:
        - mask_array: A binary mask where 1 indicates bad pixels, 0 indicates good pixels.
        """
        mask_array = np.zeros(quality_data.shape, dtype=bool)

        # Replace NaNs with 0 and convert to integer
        quality_data = np.nan_to_num(quality_data, nan=0).astype(np.int8)

        # Iterate through the bits to generate the mask
        for bit in bit_nums:
            mask_temp = (quality_data & (1 << bit)) > 0
            mask_array = np.logical_or(mask_array, mask_temp)

        return mask_array

    def compute_selected_indices(
        self,
        red_band_file,
        nir_band_file,
        green_band_file,
        blue_band_file,
        fmask_file,
        output_dir,
        selected_indices,
        swir_band_file=None,
        red_edge_band_file=None,
    ):
        """
        Compute selected vegetation indices like NDVI, GCVI, EVI, SAVI, etc., apply scaling, and QA masking.
        Avoid recomputation if the file already exists.

        Parameters:
        - red_band_file, nir_band_file, green_band_file, blue_band_file, fmask_file: Paths to the necessary bands
        - output_dir: Directory to save the output indices
        - selected_indices: A list of selected indices to compute (e.g., ['NDVI', 'EVI', 'SAVI'])
        - swir_band_file, red_edge_band_file: Optional paths to the SWIR and Red Edge bands for specific indices
        """

        # Ensure the output directory exists
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Open the bands with scaling
        scale_factor = 0.0001
        red_band = rxr.open_rasterio(red_band_file).squeeze() * scale_factor
        nir_band = rxr.open_rasterio(nir_band_file).squeeze() * scale_factor
        green_band = rxr.open_rasterio(green_band_file).squeeze() * scale_factor
        blue_band = rxr.open_rasterio(blue_band_file).squeeze() * scale_factor

        # Open the QA mask (Fmask)
        fmask = rxr.open_rasterio(fmask_file).squeeze()

        # Create the quality mask from Fmask using the create_quality_mask function
        bit_nums = [
            1,
            2,
            3,
            4,
            5,
        ]  # Define which bits to use for masking (can be adjusted)
        mask_layer = self.create_quality_mask(fmask.data, bit_nums)

        # Apply the QA mask to each band (good pixels are where mask_layer == False)
        red_band = red_band.where(~mask_layer)
        nir_band = nir_band.where(~mask_layer)
        green_band = green_band.where(~mask_layer)
        blue_band = blue_band.where(~mask_layer)

        # Available index computation functions
        index_functions = {
            "NDVI": self.compute_ndvi,
            "GCVI": self.compute_gcvi,
            "EVI": self.compute_evi,
            "SAVI": self.compute_savi,
            "MSAVI": self.compute_msavi,
            "NDWI": self.compute_ndwi,
            "GNDVI": self.compute_gndvi,
            "ARVI": self.compute_arvi,
            "NDMI": self.compute_ndmi if swir_band_file else None,
            "RENDVI": self.compute_rendvi if red_edge_band_file else None,
            "VARI": self.compute_vari,
        }

        # Loop over the selected indices and compute them if not already saved
        for index in tqdm(selected_indices, desc="Computing indices"):
            output_file = output_dir / f"{index}.tif"
            if output_file.exists():
                print(f"{index} already exists, skipping computation.")
                continue

            index_function = index_functions.get(index)
            if index_function is not None:
                # Prepare the arguments based on the index
                if index == "NDMI":
                    index_function(nir_band, output_file, swir_band_file)
                elif index == "RENDVI":
                    index_function(nir_band, output_file, red_edge_band_file)
                else:
                    index_function(
                        red_band, nir_band, green_band, blue_band, output_file
                    )
            else:
                print(
                    f"{index} is not available or missing necessary bands (e.g., SWIR or Red Edge)."
                )

    # Functions to compute individual indices
    def compute_ndvi(self, red_band, nir_band, *args):
        output_file = args[-1]
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        ndvi.rio.to_raster(output_file)

    def compute_gcvi(self, red_band, nir_band, green_band, *args):
        output_file = args[-1]
        gcvi = (nir_band / green_band) - 1
        gcvi.rio.to_raster(output_file)

    def compute_evi(self, red_band, nir_band, green_band, blue_band, *args):
        output_file = args[-1]
        evi = (
            2.5
            * (nir_band - red_band)
            / (nir_band + 6 * red_band - 7.5 * blue_band + 1)
        )
        evi.rio.to_raster(output_file)

    def compute_savi(self, red_band, nir_band, *args):
        output_file = args[-1]
        L = 0.5
        savi = ((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L)
        savi.rio.to_raster(output_file)

    def compute_msavi(self, red_band, nir_band, *args):
        output_file = args[-1]
        msavi = (
            2 * nir_band
            + 1
            - np.sqrt((2 * nir_band + 1) ** 2 - 8 * (nir_band - red_band))
        ) / 2
        msavi.rio.to_raster(output_file)

    def compute_ndwi(self, red_band, nir_band, green_band, *args):
        output_file = args[-1]
        ndwi = (green_band - nir_band) / (green_band + nir_band)
        ndwi.rio.to_raster(output_file)

    def compute_gndvi(self, red_band, nir_band, green_band, *args):
        output_file = args[-1]
        gndvi = (nir_band - green_band) / (nir_band + green_band)
        gndvi.rio.to_raster(output_file)

    def compute_arvi(self, red_band, nir_band, green_band, blue_band, *args):
        output_file = args[-1]
        arvi = (nir_band - (2 * red_band - blue_band)) / (
            nir_band + (2 * red_band - blue_band)
        )
        arvi.rio.to_raster(output_file)

    def compute_ndmi(self, nir_band, output_file, swir_band_file, *args):
        swir_band = rxr.open_rasterio(swir_band_file).squeeze() * 0.0001
        ndmi = (nir_band - swir_band) / (nir_band + swir_band)
        ndmi.rio.to_raster(output_file)

    def compute_rendvi(self, nir_band, output_file, red_edge_band_file, *args):
        red_edge_band = rxr.open_rasterio(red_edge_band_file).squeeze() * 0.0001
        rendvi = (nir_band - red_edge_band) / (nir_band + red_edge_band)
        rendvi.rio.to_raster(output_file)

    def compute_vari(self, red_band, nir_band, green_band, blue_band, *args):
        output_file = args[-1]
        vari = (green_band - red_band) / (green_band + red_band - blue_band)
        vari.rio.to_raster(output_file)
