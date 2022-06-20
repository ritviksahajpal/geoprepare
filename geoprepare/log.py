import os
import pdb
import logging
import logzero
import arrow as ar

from pathlib import Path

logging.getLogger('chardet.charsetprober').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Logger:
    # adapted from https://gist.github.com/empr/2036153
    # Level	    Numeric value
    # CRITICAL	      50
    # ERROR	          40
    # WARNING	      30
    # INFO	          20
    # DEBUG	          10
    # NOTSET	       0
    def __init__(self, dir_log, name_project='geoprepare', name_fl='logger', level=logging.INFO):
        log_format = '[%(asctime)s] %(message)s'
        dir_log = Path(dir_log) / name_project / ar.now().format('MMMM_DD_YYYY')
        os.makedirs(dir_log, exist_ok=True)

        name_fl = name_fl + '.txt'
        self.logger = logzero.setup_logger(name=name_fl,
                                           logfile=dir_log / name_fl,
                                           formatter=logzero.LogFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M"),
                                           maxBytes=1e6,   # 1 MB size
                                           backupCount=3,
                                           level=level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
