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


class GeoExtract(base.BaseGeo):
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
        self.forecast_seasons = ast.literal_eval(self.parser.getint('DEFAULT', 'forecast_seasons'))


def run(path_config_file='geoextract.txt'):
    from .extract import extract as obj

    # Read in configuration file
    geoextract = GeoExtract(path_config_file)

    # Run the extraction process
    obj.run(geoextract)


if __name__ =='__main__':
    run()
