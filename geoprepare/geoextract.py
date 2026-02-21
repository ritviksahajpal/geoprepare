"""
geoextract.py - Extract EO variable statistics per admin region.

Orchestrates the spatial extraction of Earth Observation (EO) data
(NDVI, CHIRPS, CPC, ESI, etc.) for each country, crop, and admin region.
Delegates the heavy lifting to extract/extract_EO.py and optionally
generates EWCM region-assignment plots via georegion.py.

Pipeline: download (geodownload.py) -> extract (geoextract.py) -> merge (geomerge.py)
"""
import os
import ast
import datetime

from pathlib import Path

from . import base


class GeoExtract(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section="DEFAULT"):
        """

        Args:
            section ():

        Returns:

        """
        self.project_name = self.parser.get("DEFAULT", "project_name")
        super().parse_config(self.project_name, section="DEFAULT")
        self.method = self.parser.get("DEFAULT", "method")
        self.countries = ast.literal_eval(self.parser.get("DEFAULT", "countries"))

        self.redo = self.parser.getboolean("DEFAULT", "redo")
        self.parallel_extract = self.parser.getboolean("PROJECT", "parallel_extract")


def run(path_config_file="geoextract.txt"):
    # Read in configuration file.
    obj = GeoExtract(path_config_file)
    obj.parse_config("DEFAULT")

    from . import utils
    eo_vars = ast.literal_eval(obj.parser.get(obj.countries[0], "eo_model"))
    num_cpus = int(obj.fraction_cpus * os.cpu_count()) if obj.parallel_extract else 1
    utils.display_run_summary("GeoExtract Runner", [
        ("Countries", obj.countries),
        ("Years", f"{obj.start_year} - {obj.end_year}"),
        ("EO vars", eo_vars),
        ("Method", obj.method),
        ("Redo", str(obj.redo)),
        ("Parallel", str(obj.parallel_extract)),
        ("CPUs", str(num_cpus)),
        ("Intermed dir", str(obj.dir_intermed)),
        ("Output dir", str(obj.dir_output)),
    ])

    from .extract import extract_EO as ee

    ee.run(obj)

    # Generate region-assignment plots for all EWCM countries
    from . import georegion

    dir_plots = obj.dir_output / "region_plots"
    georegion.plot_all_countries(
        path_config_file=path_config_file,
        dir_boundary_files=obj.dir_boundary_files,
        dir_output=dir_plots,
        dir_cache=obj.dir_intermed / "region_cache",
        redo=obj.redo,
    )


if __name__ == "__main__":
    run()