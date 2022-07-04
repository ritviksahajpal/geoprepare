###############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
# July 4, 2022
###############################################################################
import os
import pdb
import ast

from pathlib import Path
from . import common


class BaseGeo:
    def __init__(self, path_config_file=['geoprepare.txt', 'geoextract.txt']):
        self.parser = common.read_config(path_config_file)
        self.redo_last_year = True

    def pp_config(self, section='DEFAULT'):
        from pprint import pformat

        if not self.parser:
            raise ValueError('Parser not initialized')
        else:
            self.logger.info(f'Downloading {section}')
            self.logger.info(pformat(dict(self.parser[section])))

    def parse_config(self, section='DEFAULT'):
        """

        Args:
            section ():

        Returns:

        """
        self.dir_base = Path(self.parser.get('PATHS', 'dir_base'))
        self.dir_log = Path(self.parser.get('PATHS', 'dir_log'))
        self.dir_input = Path(self.parser.get('PATHS', 'dir_input'))
        self.dir_interim = Path(self.parser.get('PATHS', 'dir_interim'))
        self.dir_download = Path(self.parser.get('PATHS', 'dir_download'))
        self.dir_output = Path(self.parser.get('PATHS', 'dir_output'))

        self.parallel_process = self.parser.getboolean(section, 'parallel_process')
        self.start_year = self.parser.getint(section, 'start_year')
        self.end_year = self.parser.getint(section, 'end_year')
        self.fraction_cpus = self.parser.getfloat(section, 'fraction_cpus')
