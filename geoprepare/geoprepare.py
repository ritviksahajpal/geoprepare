import os
import pdb

from configparser import ConfigParser, ExtendedInterpolation


def setup_config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', default='config.txt')
    args = parser.parse_args()

    parser = ConfigParser(inline_comment_prefixes=(';',), interpolation=ExtendedInterpolation())
    parser.read(args.config)

    return args, parser


if __name__ == '__main__':
    # Read in configuration file
    args, parser = setup_config()

    # Execute based on configuration file

    pass
