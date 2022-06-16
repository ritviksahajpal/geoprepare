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
        if not self.parser:
            raise ValueError('Parser not initialized')
        else:
            self.logger.info(dict(self.parser[section]))

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
            import datasets.CHIRPS as CHIRPS

            # Parse configuration file for CHIRPS
            geoprep.parse_config('CHIRPS')
            # Print all elements of configuration file
            geoprep.pp_config('CHIRPS')

            geoprep.fill_value = geoprep.parser.getint('CHIRPS', 'fill_value')
            geoprep.prelim = geoprep.parser.get('CHIRPS', 'prelim')
            geoprep.final = geoprep.parser.get('CHIRPS', 'final')

            CHIRPS.run(geoprep)
        elif dataset == 'NDVI':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'AGERA5':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'CHIRPS-GEFS':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'FLDAS':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'LST':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'ESI':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'CPC':
            raise NotImplementedError(f'{dataset} not implemented')
        elif dataset == 'SOIL-MOISTURE':
            raise NotImplementedError(f'{dataset} not implemented')
        else:
            raise ValueError(f'{dataset} not implemented')


if __name__ == '__main__':
    run()
