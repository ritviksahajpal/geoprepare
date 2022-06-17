###############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
###############################################################################
import os
import ast
import pdb
import datetime
import argparse

from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation

from . import log


def read_config(path_config_file='config.txt'):
    """

    Args:
        path_config_file ():

    Returns:

    """
    parser = ConfigParser(inline_comment_prefixes=(';',), interpolation=ExtendedInterpolation())

    if not os.path.isfile(path_config_file):
        raise FileNotFoundError(f'Cannot find {path_config_file}')

    try:
        parser.read(path_config_file)
    except Exception as e:
        raise IOError(f'Cannot read {path_config_file}: {e}')

    return parser


class geoprepare:
    def __init__(self, path_config_file):
        self.parser = read_config(path_config_file)
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
        self.dir_base = Path(self.parser.get('DATASETS', 'dir_base'))
        self.dir_log = Path(self.parser.get('DATASETS', 'dir_log'))
        self.dir_input = Path(self.parser.get('DATASETS', 'dir_input'))
        self.dir_interim = Path(self.parser.get('DATASETS', 'dir_interim'))
        self.dir_download = Path(self.parser.get('DATASETS', 'dir_download'))
        self.dir_output = Path(self.parser.get('DATASETS', 'dir_output'))

        self.parallel_process = self.parser.getboolean(section, 'parallel_process')
        self.start_year = self.parser.getint(section, 'start_year')
        self.end_year = self.parser.getint(section, 'end_year')
        self.fraction_cpus = self.parser.getfloat(section, 'fraction_cpus')

        # check if current date is on or after March 1st. If it is then set redo_last_year flag to False else True
        # If redo_last_year is True then we redo the download, processing etc of last year's data
        if datetime.datetime.today().month >= 3:
            self.redo_last_year = False

        # Set up logger
        self.logger = log.Logger(dir_log=self.dir_log,
                                 name_fl=self.parser.get('DEFAULT', 'logfile'))


def run(path_config_file='config.txt'):
    # Read in configuration file
    geoprep = geoprepare(path_config_file)
    datasets = ast.literal_eval(geoprep.parser.get('DATASETS', 'datasets'))

    # Loop through all datasets in parser
    for dataset in datasets:
        if dataset == 'CHIRPS':
            from .datasets import CHIRPS

            # Parse configuration file for CHIRPS
            geoprep.parse_config('CHIRPS')

            geoprep.fill_value = geoprep.parser.getint('CHIRPS', 'fill_value')
            geoprep.prelim = geoprep.parser.get('CHIRPS', 'prelim')
            geoprep.final = geoprep.parser.get('CHIRPS', 'final')

            # Print all elements of configuration file
            geoprep.pp_config('CHIRPS')

            CHIRPS.run(geoprep)
        elif dataset == 'NDVI':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'AGERA5':
            from .datasets import AgERA5

            # Parse configuration file for AGERA5
            geoprep.parse_config('AGERA5')

            # Print all elements of configuration file
            geoprep.pp_config('AGERA5')

            AgERA5.run(geoprep)
        elif dataset == 'CHIRPS-GEFS':
            from .datasets import CHIRPS_GEFS

            # Parse configuration file for CHIRPS-GEFS
            geoprep.parse_config('CHIRPS-GEFS')
            geoprep.data_dir = geoprep.parser.get('CHIRPS-GEFS', 'data_dir')
            geoprep.fill_value = geoprep.parser.getint('CHIRPS', 'fill_value')

            # Print all elements of configuration file
            geoprep.pp_config('CHIRPS-GEFS')

            CHIRPS_GEFS.run(geoprep)
        elif dataset == 'FLDAS':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'LST':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'ESI':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'CPC':
            from .datasets import CPC

            # Parse configuration file for CPC
            geoprep.parse_config('CPC')
            geoprep.data_dir = geoprep.parser.get('CPC', 'data_dir')

            # Print all elements of configuration file
            geoprep.pp_config('CPC')

            CPC.run(geoprep)
        elif dataset == 'SOIL-MOISTURE':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'AVHRR':
            from .datasets import AVHRR

            # Parse configuration file for AVHRR
            geoprep.parse_config('AVHRR')
            geoprep.data_dir = geoprep.parser.get('AVHRR', 'data_dir')

            # Print all elements of configuration file
            geoprep.pp_config('AVHRR')

            AVHRR.run(geoprep)
        else:
            raise ValueError(f'{dataset} not implemented')


if __name__ == '__main__':
    run()
