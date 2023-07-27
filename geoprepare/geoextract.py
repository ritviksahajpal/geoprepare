###############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
# July 4, 2022
###############################################################################
import os
import ast
import pdb
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
        super().parse_config(section="DEFAULT")

        self.countries = ast.literal_eval(self.parser.get("DEFAULT", "countries"))
        self.dir_masks = Path(self.parser.get("PATHS", "dir_masks"))
        self.dir_regions = Path(self.parser.get("PATHS", "dir_regions"))
        self.dir_regions_shp = Path(self.parser.get("PATHS", "dir_regions_shp"))
        self.dir_crop_masks = Path(self.parser.get("PATHS", "dir_crop_masks"))
        self.redo = self.parser.getboolean("DEFAULT", "redo")

        # self.forecast_seasons = ast.literal_eval(self.parser.getint('DEFAULT', 'forecast_seasons'))


def run(path_config_file="geoextract.txt"):
    # Read in configuration file.
    geoextract = GeoExtract(path_config_file)
    geoextract.parse_config("DEFAULT")

    # Create crop masks
    from .extract import extract_crop_masks as ec

    ec.run(geoextract)

    # Extract EO data
    from .extract import extract_EO as ee

    ee.run(geoextract)


if __name__ == "__main__":
    run()
