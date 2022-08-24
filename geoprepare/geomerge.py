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

        self.dir_models = Path(self.parser.get('PATHS', 'dir_models'))
        self.dir_model_inputs = self.dir_models / 'inputs'
        self.dir_model_outputs = self.dir_models / 'outputs'
        self.countries = ast.literal_eval(self.parser.get('DEFAULT', 'countries'))


def run(path_config_file='geoextract.txt'):
    # Read in configuration file.
    geomerge = GeoMerge(path_config_file)
    geomerge.parse_config('DEFAULT')

    # Merge EO data and crop calendars and yield information
    from .merge import merge_all as obj
    obj.run(geomerge)


if __name__ =='__main__':
    run()
