###############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
# July 4, 2022
###############################################################################
import os
import ast
import datetime

from pathlib import Path

from . import base


class GeoExtract(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section="DEFAULT"):
        """

        Args:
            section ():

        Returns:

        """
        self.project_name = self.parser.get("PROJECT", "project_name")
        super().parse_config(self.project_name, section="DEFAULT")
        self.method = self.parser.get("DEFAULT", "method")
        self.countries = ast.literal_eval(self.parser.get("DEFAULT", "countries"))
        self.dir_masks = Path(self.parser.get("PATHS", "dir_masks"))
        self.dir_regions = Path(self.parser.get("PATHS", "dir_regions"))
        self.dir_regions_shp = Path(self.parser.get("PATHS", "dir_regions_shp"))
        self.dir_crop_masks = Path(self.parser.get("PATHS", "dir_crop_masks"))
        # self.use_cropland_mask = self.parser.get(country, "use_cropland_mask")
        # self.crop_mask = self.dir_crop_masks / self.parser.
        self.redo = self.parser.getboolean("DEFAULT", "redo")
        # self.forecast_seasons = ast.literal_eval(self.parser.getint('DEFAULT', 'forecast_seasons'))
        self.parallel_extract = self.parser.getboolean("PROJECT", "parallel_extract")


def run(path_config_file="geoextract.txt"):
    # Read in configuration file.
    obj = GeoExtract(path_config_file)
    obj.parse_config("DEFAULT")

    from .extract import extract_EO as ee

    ee.run(obj)

    # Generate region-assignment plots for all EWCM countries
    from . import georegion

    dir_plots = obj.dir_output / "region_plots"
    georegion.plot_all_countries(
        path_config_file=path_config_file,
        dir_regions_shp=obj.dir_regions_shp,
        dir_regions=obj.dir_regions,
        dir_output=dir_plots,
        dir_cache=obj.dir_interim / "region_cache",
        redo=obj.redo,
    )


if __name__ == "__main__":
    run()