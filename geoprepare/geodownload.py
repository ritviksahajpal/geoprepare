##############################################################################
# Ritvik Sahajpal
# email: ritvik@umd.edu
# June 18, 2022
###############################################################################
import ast
import datetime

from tqdm import tqdm

from . import base


class GeoDownload(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section="DEFAULT"):
        """

        Args:
            section ():

        Returns:

        """
        super().parse_config(section="DEFAULT")

        # check if current date is on or after March     1st. If it is then set redo_last_year flag to False else True
        # If redo_last_year is True then we redo the download, processing of last year's data
        self.redo_last_year = False if datetime.datetime.today().month >= 3 else True


def run(path_config_file=["geobase.txt"]):
    # Read in configuration file
    geoprep = GeoDownload(path_config_file)
    datasets = ast.literal_eval(geoprep.parser.get("DATASETS", "datasets"))

    # Loop through all datasets in parser
    pbar = tqdm(datasets, desc="Downloading Datasets")
    for dataset in datasets:
        pbar.set_description(f"Downloading {dataset}")
        pbar.update()

        if dataset == "CHIRPS":
            from .datasets import CHIRPS as obj

            geoprep.fill_value = geoprep.parser.getint("CHIRPS", "fill_value")
            # Get CHIRPS version (v2 or v3), default to v2 for backward compatibility
            geoprep.version = geoprep.parser.get("CHIRPS", "version", fallback="v2")
            # Get disaggregation method for v3 (sat or rnl), default to sat
            geoprep.disagg = geoprep.parser.get("CHIRPS", "disagg", fallback="sat")
            # Legacy paths (kept for reference but not used in new code)
            geoprep.prelim = geoprep.parser.get("CHIRPS", "prelim")
            geoprep.final = geoprep.parser.get("CHIRPS", "final")
        elif dataset in ["NDVI", "VIIRS"]:
            from .datasets import NDVI as obj

            # product vi start_year scale_glam scale_mark print_missing
            geoprep.product = geoprep.parser.get(dataset, "product")
            geoprep.vi = geoprep.parser.get(dataset, "vi")
            geoprep.start_year = geoprep.parser.getint(dataset, "start_year")
            # Use GLAM scaling NDVI * 10000
            geoprep.scale_glam = geoprep.parser.getboolean(dataset, "scale_glam")
            # Use Mark's scaling (NDVI * 200) + 50
            geoprep.scale_mark = geoprep.parser.getboolean(dataset, "scale_mark")
            # Print missing dates (+status) and exit
            geoprep.print_missing = geoprep.parser.getboolean(dataset, "print_missing")
        elif dataset == "AGERA5":
            from .datasets import AgERA5 as obj

            geoprep.variables = ast.literal_eval(
                geoprep.parser.get("AGERA5", "variables")
            )
        elif dataset == "CHIRPS-GEFS":
            from .datasets import CHIRPS_GEFS as obj

            geoprep.data_dir = geoprep.parser.get("CHIRPS-GEFS", "data_dir")
            geoprep.fill_value = geoprep.parser.getint("CHIRPS", "fill_value")
        elif dataset == "LST":
            from .datasets import LST as obj

            geoprep.num_update_days = geoprep.parser.getint("LST", "num_update_days")
        elif dataset == "ESI":
            from .datasets import ESI as obj

            geoprep.data_dir = geoprep.parser.get("ESI", "data_dir")
            geoprep.list_products = ast.literal_eval(geoprep.parser.get("ESI", "list_products"))
        elif dataset == "CPC":
            from .datasets import CPC as obj

            geoprep.data_dir = geoprep.parser.get("CPC", "data_dir")
        elif dataset == "SOIL-MOISTURE":
            from .datasets import Soil_Moisture as obj

            geoprep.data_dir = geoprep.parser.get("SOIL-MOISTURE", "data_dir")
        elif dataset == "NSIDC":
            from .datasets import NSIDC as obj
        elif dataset == "AVHRR":
            from .datasets import AVHRR as obj

            geoprep.data_dir = geoprep.parser.get("AVHRR", "data_dir")
        elif dataset == "VHI":
            from .datasets import VHI as obj

            geoprep.url_historic = geoprep.parser.get("VHI", "data_historic")
            geoprep.url_current = geoprep.parser.get("VHI", "data_current")
        elif dataset == "FPAR":
            from .datasets import FPAR as obj

            geoprep.data_dir = geoprep.parser.get("FPAR", "data_dir")
        elif dataset == "FLDAS":
            from .datasets import FLDAS as obj

            # FLDAS configuration
            # Whether to use NMME with SPEAR model (default: False -> uses NMME_noSPEAR)
            geoprep.fldas_use_spear = geoprep.parser.getboolean(
                "FLDAS", "use_spear", fallback=False
            )
            # Data types to download: forecast, openloop, or both
            geoprep.fldas_data_types = ast.literal_eval(
                geoprep.parser.get("FLDAS", "data_types", fallback="['forecast']")
            )
            # Variables to extract from NetCDF files
            geoprep.fldas_variables = ast.literal_eval(
                geoprep.parser.get(
                    "FLDAS",
                    "variables",
                    fallback="['SoilMoist_tavg', 'TotalPrecip_tavg', 'Tair_tavg', 'Evap_tavg', 'TWS_tavg']"
                )
            )
            # Forecast lead times to process (0-5)
            geoprep.fldas_leads = ast.literal_eval(
                geoprep.parser.get("FLDAS", "leads", fallback="[0, 1, 2, 3, 4, 5]")
            )
            # Whether to compute anomalies
            geoprep.fldas_compute_anomalies = geoprep.parser.getboolean(
                "FLDAS", "compute_anomalies", fallback=False
            )
        else:
            raise ValueError(f"{dataset} not implemented")

        # Parse configuration file
        geoprep.parse_config(dataset)
        # Print all elements of configuration file
        geoprep.pp_config(dataset)
        # Execute!
        obj.run(geoprep)


if __name__ == "__main__":
    run()