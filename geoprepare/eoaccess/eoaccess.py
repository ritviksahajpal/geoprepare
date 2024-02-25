import os
import logging
import rasterio as rio
import xarray as xr
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

        # Create output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create logging directory if it does not exist
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

        # Convert shapefile to EPSG:4326 if not already
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

        print(f" Using {type(fileset[0])} filesystem")
        # Open each file with rioxarray and store in a list
        datasets = [rxr.open_rasterio(file) for file in fileset]

        # Concatenate datasets with a progress bar
        with tqdm(total=len(datasets), desc="Concatenating datasets") as pbar:
            def concat_with_progress(ds_list):
                for ds in ds_list:
                    yield ds
                    pbar.update(1)

            merged_dataset = xr.concat(concat_with_progress(datasets), dim='new_dimension')
            breakpoint()

        # Merge the datasets
        # This example uses concat along a new dimension; adjust as needed
        merged_dataset = xr.concat(datasets)

        breakpoint()
        ds = xr.open_mfdataset(fileset)
        ds
        breakpoint()

    @staticmethod
    def download(item):
        result, output_dir = item
        ea.download([result], str(output_dir))

    def download_parallel(self):
        # Create a list of tuples containing result and output_dir
        combinations = [(result, self.output_dir) for result in self.results]

        num_cpu = int(cpu_count() * 0.6)
        with Pool(num_cpu) as p:
            for i, _ in enumerate(p.imap_unordered(self.download, combinations)):
                pass

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
        # Both bbox and shapefile cannot be not None at the same time
        if self.bbox and self.shapefile:
            raise ValueError("Both bbox and shapefile cannot be specified")

        # if shapefile exists then it should have columns specifying start and end dates
        # if self.shapefile:
        #     assert self.start_date_col and self.end_date_col, "Start and end date columns must be specified"
        #     assert self.start_date_col in self.shapefile.columns, f"{self.start_date_col} not found in shapefile"
        #     assert self.end_date_col in self.shapefile.columns, f"{self.end_date_col} not found in shapefile"

        # Create mosaic directory within output directory
        self.mosaic_dir = Path(os.path.join(self.input_dir, "mosaic"))
        os.makedirs(self.mosaic_dir, exist_ok=True)

    def get_ts(self):
        # Loop over shapefile
        dg = gpd.read_file(self.shapefile)
        for index, row in tqdm(dg.iterrows(), desc="Getting time-series"):
            breakpoint()
            pass

    def mosaic(self):
        # Check if self.dataset contains either HLSS30 or HLSL30
        if "HLSS30" in self.dataset or "HLSL30" in self.dataset:
            def group_files_by_band_and_date():
                grouped_files = defaultdict(list)

                # Iterate through all files in the directory
                for filename in os.listdir(self.input_dir):
                    if filename.endswith(".tif"):
                        # Parse filename to extract band and date
                        parts = filename.split(".")
                        band = parts[-2]  # Spectral band is the second last part
                        date = parts[3][:-7]  # Julian Date of Acquisition

                        # Group the files
                        grouped_files[(band, date)].append(filename)

                return grouped_files

            grouped_files = group_files_by_band_and_date()

            pbar = tqdm(grouped_files.items())
            for key, files in pbar:
                band, date = key

                pbar.set_description(f"Mosaicing: {band} {date}")
                pbar.update()

                mosaic_file = self.mosaic_dir / f"mosaic_{band}_{date}.tif"
                if os.path.exists(mosaic_file):
                    continue

                tif_files = [Path(self.input_dir) / filename for filename in files]
                utils.mosaic(tif_files, mosaic_file)
