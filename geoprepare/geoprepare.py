import os
import pdb

from pathlib import Path

def read_config():
    """

    Returns:

    """
    import argparse
    from configparser import ConfigParser, ExtendedInterpolation

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', default='config.txt', help='Path to configuration file')
    args = parser.parse_args()

    parser = ConfigParser(inline_comment_prefixes=(';',), interpolation=ExtendedInterpolation())

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'Cannot find {args.config}')

    try:
        parser.read(args.config)
    except Exception as e:
        raise IOError(f'Cannot read {args.config}: {e}')

    return args, parser


class geoprepare:
    def __init__(self):
        """

        Args:
            args (object):
            parser (object):
        """
        self.args, self.parser = read_config()

    def pp_config(self):
        """

        Returns:

        """
        for section in self.parser.sections():
            print(section, dict(self.parser[section]))

    def parse_config(self, section='DEFAULT'):
        """

        Returns:

        """
        self.dir_base = Path(self.parser.get('DATASETS', 'dir_base'))
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
        import datetime

        self.redo_last_year = True
        if datetime.datetime.today().month >= 3:
            self.redo_last_year = False


if __name__ == '__main__':
    # Read in configuration file
    geoprep = geoprepare()

    # Print all elements of configuration file
    geoprep.pp_config()

    # Loop through all sections in parser
    for section in geoprep.parser.sections():
        if section == 'CHIRPS':
            import datasets.CHIRPS as CHIRPS

            # Parse configuration file for CHIRPS
            geoprep.parse_config('CHIRPS')
            geoprep.fill_value = geoprep.parser.getint('CHIRPS', 'fill_value')
            geoprep.prelim = geoprep.parser.get('CHIRPS', 'prelim')
            geoprep.final = geoprep.parser.get('CHIRPS', 'final')

            CHIRPS.run(geoprep)

    pdb.set_trace()
    pass
