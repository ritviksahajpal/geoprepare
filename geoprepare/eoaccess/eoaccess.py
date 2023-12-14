import os
import logging
import rasterio
from pathlib import Path
import earthaccess as ea
import geopandas as gpd
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from rasterio.merge import merge

from .. import log


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

        try:
            ea.login(strategy=self.login_strategy)
        except ea.exceptions.LoginError:
            ea.login(strategy="interactive", persist=True)

        # Change output_dir to Path if it is a string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Create output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create logging directory if it does not exist
        self.logger = log.Logger(
            dir_log=self.output_dir / "logs",
            name_project=self.logging_project,
            name_fl=self.logging_file,
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

    def download(self):
        files = ea.download(self.results, self.output_dir)

        self.logger.info(f"Downloading {len(files)} files")


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

                # Read the tif files
                src_files_to_mosaic = []
                for fp in tif_files:
                    src = rasterio.open(fp)
                    src_files_to_mosaic.append(src)

                # Mosaic the files
                mosaic, out_trans = merge(src_files_to_mosaic)

                # Copy the metadata
                out_meta = src.meta.copy()

                # Update the metadata
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                    }
                )

                # Write the mosaic raster to disk
                with rasterio.open(mosaic_file, "w", **out_meta) as dest:
                    dest.write(mosaic)

                # Close the files
                for src in src_files_to_mosaic:
                    src.close()


def main():
    dg = gpd.read_file(r"D:\Users\ritvik\projects\GEOGLAM\Input\SARA\field_bound_shp\compiled.shp", engine="pyogrio")

    # Convert to CRS 4326 if not already
    if dg.crs != "EPSG:4326":
        dg = dg.to_crs("EPSG:4326")

    # Iterate over each row of the shapefile
    for index, row in tqdm(dg.iterrows(), desc="Iterating over shapefile", total=len(dg)):
        # Get bbox from geometry of the row
        bbox = row.geometry.bounds

        obj = NASAEarthAccess(
            dataset=["HLSL30", "HLSS30"],
            bbox=bbox,
            temporal=(f"{row['year']}-01-01", f"{row['year']}-12-31"),
            output_dir=r"D:\Users\ritvik\projects\GEOGLAM\Input\HLS\SARA",
        )

        obj.search_data()
        if obj.results:
            obj.download()

    obj = EarthAccessProcessor(
        dataset=["HLSL30", "HLSS30"],
        input_dir=r"D:\Users\ritvik\projects\GEOGLAM\Input\HLS",
        shapefile=Path(
            r"D:\Users\ritvik\projects\GEOGLAM\Input\SARA\field_bound_shp\compiled.shp"
        ),
    )
    obj.mosaic()

