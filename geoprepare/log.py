import os
import logging
import logzero
import arrow as ar
from pathlib import Path


class Logger:
    # adapted from https://gist.github.com/empr/2036153
    # Level	    Numeric value
    # CRITICAL	      50
    # ERROR	          40
    # WARNING	      30
    # INFO	          20
    # DEBUG	          10
    # NOTSET	       0
    def __init__(
        self,
        dir_log,  # Path to the directory where the log file will be saved
        project="geoprepare",  # Name of the project, this will be created as a subdirectory in dir_log
        file="logger.txt",  # Name of the log file
        level=logging.INFO,  # Logging level (see above)
    ):
        log_format = "[%(asctime)s] %(message)s"
        dir_log = Path(dir_log) / project / ar.now().format("MMMM_DD_YYYY")
        os.makedirs(dir_log, exist_ok=True)

        self.logger = logzero.setup_logger(
            name=file,
            logfile=dir_log / file,
            formatter=logzero.LogFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M"),
            maxBytes=int(1e6),  # 1 MB size
            backupCount=3,
            level=level,
        )

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
