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

from . import log
from . import common
from . import base


class GeoMerge(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section='DEFAULT'):
        """

        Args:
            section ():

        Returns:

        """
        super().parse_config(section='DEFAULT')

        self.countries = ast.literal_eval(self.parser.get('DEFAULT', 'countries'))
        self.dir_masks = Path(self.parser.get('PATHS', 'dir_masks'))
        self.dir_regions = Path(self.parser.get('PATHS', 'dir_regions'))
        self.dir_regions_shp = Path(self.parser.get('PATHS', 'dir_regions_shp'))
        self.dir_crop_masks = Path(self.parser.get('PATHS', 'dir_crop_masks'))
        self.redo = self.parser.getboolean('DEFAULT', 'redo')

        # self.forecast_seasons = ast.literal_eval(self.parser.getint('DEFAULT', 'forecast_seasons'))


def run(path_config_file='geoextract.txt'):
    # Read in configuration file.
    geomerge = GeoMerge(path_config_file)
    geomerge.parse_config('DEFAULT')

    #
    # Merge EO data and crop calendars and yield information
    from .merge import merge_all as obj
    obj.run(geomerge)


if __name__ =='__main__':
    run()
