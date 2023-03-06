###############################################################################
# Ritvik Sahajpal, Tarun Murali
# email: ritvik@umd.edu
# July 4, 2022
###############################################################################
import os

from pathlib import Path

from . import utils
from . import log

class BaseGeo:
    def __init__(self, path_config_file=['geoprepare.txt', 'geoextract.txt']):
        self.parser = utils.read_config(path_config_file)
        self.redo_last_year = True

        # Dictionary of crop growth stages
        self.dict_growth_stages = {1: 'Planting - Early Vegetative', 2: 'Vegetative - Reproductive', 3: 'Ripening through Harvest'}

        # Dictionary of crop conditions
        self.dict_crop_conditions = {1: 'Poor', 2: 'Watch', 3: 'Favourable', 4: 'Exceptional'}

        # Dictionary of crop condition trends
        self.dict_trends = {1: 'Declining', 2: 'Stable', 3: 'Improving'}

    def pp_config(self, section='DEFAULT'):
        from pprint import pformat

        if not self.parser:
            raise ValueError('Parser not initialized')
        else:
            self.logger.info(f'Downloading {section}')
            self.logger.info(pformat(dict(self.parser[section])))

    def compute_logging_level(self):
        level = 10

        if self.logging_level == 'DEBUG':
            level = 10
        elif self.logging_level == 'INFO':
            level = 20
        elif self.logging_level == 'WARNING':
            level = 30
        elif self.logging_level == 'ERROR':
            level = 40
        elif self.logging_level == 'CRITICAL':
            level = 50
        else:
            raise ValueError(f'Invalid logging level {self.logging_level}')

        return level

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
        self.dir_global_datasets = Path(self.parser.get('PATHS', 'dir_global_datasets'))
        self.logging_level= self.parser.get('LOGGING', 'level')
        self.parallel_process = self.parser.getboolean(section, 'parallel_process')
        self.start_year = self.parser.getint(section, 'start_year')
        self.end_year = self.parser.getint(section, 'end_year')
        self.fraction_cpus = self.parser.getfloat(section, 'fraction_cpus')

        level = self.compute_logging_level()

        # Set up logger
        self.logger = log.Logger(dir_log=self.dir_log,
                                 name_fl=self.parser.get('DEFAULT', 'logfile'),
                                 level=level)
