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
        dir_logs,  # Path to the directory where the log file will be saved
        project="geoprepare",  # Name of the project, this will be created as a subdirectory in dir_logs
        file="logger.txt",  # Name of the log file
        level=logging.INFO,  # Logging level (see above)
    ):
        log_format = "[%(asctime)s] %(message)s"
        dir_logs = Path(dir_logs) / project / ar.now().format("MMMM_DD_YYYY")
        os.makedirs(dir_logs, exist_ok=True)

        self.logger = logzero.setup_logger(
            name=file,
            logfile=dir_logs / file,
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


class SafeLogger:
    """
    A multiprocessing-safe logger that uses stdlib logging instead of logzero.

    logzero's RotatingFileHandler is not safe across multiple processes —
    concurrent workers can corrupt the log file or lose messages.  This
    class writes to stderr via StreamHandler, which is process-safe.

    It is also pickle-safe (implements __getstate__/__setstate__) so it
    can be sent to multiprocessing workers.
    """

    _LOG_FORMAT = "[%(asctime)s] %(message)s"

    def __init__(self, name="geoprepare", level=logging.INFO):
        self._name = name
        self._level = level
        self._setup()

    def _setup(self):
        self._logger = logging.getLogger(self._name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(self._LOG_FORMAT))
            self._logger.addHandler(handler)
        self._logger.setLevel(self._level)

    def __getstate__(self):
        return {"_name": self._name, "_level": self._level}

    def __setstate__(self, state):
        self._name = state["_name"]
        self._level = state["_level"]
        self._setup()

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)


def swap_for_parallel(params):
    """
    Replace a logzero-based Logger with a process-safe SafeLogger.

    Call before spawning workers; restore the original logger afterwards.

    Returns
    -------
    Logger
        The original logger, so it can be restored after workers finish.
    """
    original_logger = params.logger
    params.logger = SafeLogger(level=params.compute_logging_level())
    return original_logger