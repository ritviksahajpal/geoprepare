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
from . import log
from logging import CRITICAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET

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
        self.dir_global_datasets = Path(self.parser.get('PATHS', 'dir_global_datasets'))
        self.logging_level= self.parser.get('LOGGING', 'level')
        self.parallel_process = self.parser.getboolean(section, 'parallel_process')
        self.start_year = self.parser.getint(section, 'start_year')
        self.end_year = self.parser.getint(section, 'end_year')
        self.fraction_cpus = self.parser.getfloat(section, 'fraction_cpus')
        level_returned = 10
        if self.logging_level=='DEBUG':
            level_returned = 10
        elif self.logging_level=='INFO':
            level_returned = 20
        elif self.logging_level=='WARNING':
            level_returned = 30
        elif self.logging_level=='ERROR':
            level_returned = 40
        elif self.logging_level =='CRITICAL':
            level_returned = 50
        # Set up logger
        self.logger = log.Logger(dir_log=self.dir_log,
                                 name_fl=self.parser.get('DEFAULT', 'logfile'),level=level_returned)
