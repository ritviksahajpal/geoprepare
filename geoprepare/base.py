###############################################################################
# Ritvik Sahajpal, Tarun Murali
# email: ritvik@umd.edu
# July 4, 2022
###############################################################################
import os

import pandas as pd
from pathlib import Path

from . import utils
from . import log

from importlib.metadata import version
print(f"Running geoprepare: {version('geoprepare')}")


class BaseGeo:
    def __init__(self, path_config_file=["geobase.txt", "geoextract.txt"]):
        self.parser = utils.read_config(path_config_file)
        self.redo_last_year = True

        # Dictionary of crop growth stages
        self.dict_growth_stages = {
            1: "Planting - Early Vegetative",
            2: "Vegetative - Reproductive",
            3: "Ripening through Harvest",
        }

        # Dictionary of crop conditions
        self.dict_crop_conditions = {
            1: "Poor",
            2: "Watch",
            3: "Favourable",
            4: "Exceptional",
        }

        # Dictionary of crop condition trends
        self.dict_trends = {1: "Declining", 2: "Stable", 3: "Improving"}

    def pp_config(self, section="DEFAULT"):
        from pprint import pformat

        if not self.parser:
            raise ValueError("Parser not initialized")
        else:
            self.logger.info(f"Downloading {section}")
            self.logger.info(pformat(dict(self.parser[section])))

    def compute_logging_level(self):
        level = 10

        if self.logging_level == "DEBUG":
            level = 10
        elif self.logging_level == "INFO":
            level = 20
        elif self.logging_level == "WARNING":
            level = 30
        elif self.logging_level == "ERROR":
            level = 40
        elif self.logging_level == "CRITICAL":
            level = 50
        else:
            raise ValueError(f"Invalid logging level {self.logging_level}")

        return level

    def parse_config(self, project_name="", section="DEFAULT"):
        """

        Args:
            project_name ():
            section ():

        Returns:

        """
        self.dir_base = Path(self.parser.get("PATHS", "dir_base"))
        self.dir_log = Path(self.parser.get("PATHS", "dir_log")) / project_name
        self.dir_input = Path(self.parser.get("PATHS", "dir_input"))
        self.dir_interim = Path(self.parser.get("PATHS", "dir_interim"))
        self.dir_download = Path(self.parser.get("PATHS", "dir_download"))
        self.dir_output = Path(self.parser.get("PATHS", "dir_output")) / project_name
        self.dir_global_datasets = Path(self.parser.get("PATHS", "dir_global_datasets"))
        self.dir_metadata = Path(self.parser.get("PATHS", "dir_metadata"))
        self.logging_level = self.parser.get("LOGGING", "level")
        self.parallel_process = self.parser.getboolean(section, "parallel_process")
        self.start_year = self.parser.getint(section, "start_year")
        self.end_year = self.parser.getint(section, "end_year")
        self.fraction_cpus = self.parser.getfloat(section, "fraction_cpus")

        level = self.compute_logging_level()

        # Set up logger
        self.logger = log.Logger(
            dir_log=self.dir_log,
            file=self.parser.get("DEFAULT", "logfile"),
            level=level,
        )

    def get_dirname(self, country):
        """

        Args:
            country ():

        Returns:

        """
        self.threshold = self.parser.getboolean(country, "threshold")

        limit_type = "floor" if self.threshold else "ceil"
        self.limit = self.parser.getint(country, limit_type)

        self.dir_threshold = (
            f"crop_t{self.limit}" if self.threshold else f"crop_p{self.limit}"
        )

    def get_key_or_value(self, input_item):
        """
        If input_item is a key, return its value.
        If input_item is a value, return its key.
        Otherwise, return None.
        """
        mapping = {
            "mz": "maize",
            "sb": "soybean",
            "rc": "rice",
            "sw": "spring_wheat",
            "ww": "winter_wheat",
            "ml": "millet",
            "tf": "teff",
            "sr": "sorghum"
        }

        # Check if the input item is a key in the mapping
        if input_item in mapping:
            return mapping[input_item]

        # Check if the input item is a value in the inverse mapping
        inverse_mapping = {v: k for k, v in mapping.items()}

        if input_item in inverse_mapping:
            return inverse_mapping[input_item]

        # If it's neither a key nor a value
        return None

    def get_calendar_sheet_name(self, crop, growing_season):
        sheet_name = None
        crop_name = self.get_key_or_value(crop)

        if crop in ["winter_wheat", "spring_wheat"]:
            sheet_name = crop_name
        else:
            sheet_name = f"{crop_name}_{growing_season}"

        return sheet_name

    def read_statistics(
        self,
        country,
        crop="mz",
        growing_season=1,
        read_calendar=False,
        read_statistics=False,
        read_countries=False,
        read_all=False,
    ):
        """
        Read the crop calendar and yield, area and production statistics from csv files
        Args:

        Returns:

        """
        category = self.parser.get(country, "category")

        # Get crop calendar information
        if read_calendar or read_all:
            sheet_name = self.get_calendar_sheet_name(crop, growing_season)
            self.path_calendar = (
                self.dir_input
                / "crop_calendars"
                / self.parser.get(category, "calendar_file")
            )
            self.df_calendar = (
                pd.read_excel(self.path_calendar, sheet_name=sheet_name)
                if os.path.isfile(self.path_calendar)
                else pd.DataFrame()
            )
            self.df_calendar = utils.harmonize_df(self.df_calendar)
            # Rename 'admin' to 'calendar_region' to match georegion lookup output
            if "admin" in self.df_calendar.columns:
                self.df_calendar.rename(columns={"admin": "calendar_region"}, inplace=True)

        # Get yield, area and production information
        if read_statistics or read_all:
            self.path_stats = (
                self.dir_input
                / "statistics"
                / self.parser.get(country, "statistics_file")
            )
            self.df_statistics = (
                pd.read_csv(self.path_stats)
                if os.path.isfile(self.path_stats)
                else pd.DataFrame()
            )
            self.df_statistics = utils.harmonize_df(self.df_statistics)

        # Get hemisphere and temperate/tropical zone information
        if read_countries or read_all:
            self.path_countries = (
                self.dir_input / "statistics" / self.parser.get("DEFAULT", "zone_file")
            )
            self.df_countries = (
                pd.read_csv(self.path_countries)
                if os.path.isfile(self.path_countries)
                else pd.DataFrame()
            )