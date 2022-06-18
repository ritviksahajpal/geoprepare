###############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
# June 18, 2022
###############################################################################
import os
import ast
import pdb
import datetime

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


class GeoPrepare:
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
    geoprep = GeoPrepare(path_config_file)
    datasets = ast.literal_eval(geoprep.parser.get('DATASETS', 'datasets'))

    # Loop through all datasets in parser
    for dataset in datasets:
        if dataset == 'CHIRPS':
            from .datasets import CHIRPS as obj

            geoprep.fill_value = geoprep.parser.getint('CHIRPS', 'fill_value')
            geoprep.prelim = geoprep.parser.get('CHIRPS', 'prelim')
            geoprep.final = geoprep.parser.get('CHIRPS', 'final')
        elif dataset == 'NDVI':
            from .datasets import NDVI as obj

            # product vi start_year scale_glam scale_mark print_missing
            geoprep.product = geoprep.parser.get('NDVI', 'product')
            geoprep.vi = geoprep.parser.get('NDVI', 'vi')
            geoprep.start_year = geoprep.parser.getint('NDVI', 'start_year')
            geoprep.scale_glam = geoprep.parser.getboolean('NDVI', 'scale_glam')  # Use GLAM scaling NDVI * 10000
            geoprep.scale_mark = geoprep.parser.getboolean('NDVI', 'scale_mark')  # Use Mark's scaling (NDVI * 200) + 50
            geoprep.print_missing = geoprep.parser.getboolean('NDVI', 'print_missing')  # Print missing dates (+status) and exit
        elif dataset == 'AGERA5':
            from .datasets import AgERA5 as obj
        elif dataset == 'CHIRPS-GEFS':
            from .datasets import CHIRPS_GEFS as obj

            geoprep.data_dir = geoprep.parser.get('CHIRPS-GEFS', 'data_dir')
            geoprep.fill_value = geoprep.parser.getint('CHIRPS', 'fill_value')
        elif dataset == 'LST':
            from .datasets import LST as obj

            geoprep.num_update_days = geoprep.parser.getint('LST', 'num_update_days')
        elif dataset == 'ESI':
            from .datasets import ESI as obj

            geoprep.data_dir = geoprep.parser.get('ESI', 'data_dir')
        elif dataset == 'CPC':
            from .datasets import CPC as obj

            geoprep.data_dir = geoprep.parser.get('CPC', 'data_dir')
        elif dataset == 'SOIL-MOISTURE':
            from .datasets import Soil_Moisture as obj

            geoprep.data_dir = geoprep.parser.get('SOIL-MOISTURE', 'data_dir')
        elif dataset == 'AVHRR':
            from .datasets import AVHRR as obj

            geoprep.data_dir = geoprep.parser.get('AVHRR', 'data_dir')
        elif dataset == 'FLDAS':
            raise NotImplementedError(f'{dataset} not implemented')
        else:
            raise ValueError(f'{dataset} not implemented')

        # Parse configuration file
        geoprep.parse_config(dataset)
        # Print all elements of configuration file
        geoprep.pp_config(dataset)
        # Execute!
        obj.run(geoprep)


if __name__ == '__main__':
    run()
