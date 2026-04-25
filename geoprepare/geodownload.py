"""
geodownload.py - Download global EO and climate datasets.

Iterates over configured datasets (NDVI, CHIRPS, CPC, ESI, AgERA5, etc.)
and downloads raw data files to dir_download. Each dataset module handles
its own source URL, file format, and temporal coverage.

Pipeline: download (geodownload.py) -> extract (geoextract.py) -> merge (geomerge.py)
"""
import ast
import re
import datetime
from pathlib import Path

from tqdm import tqdm

from . import base
from . import utils


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
    geoprep.parse_config("DEFAULT")
    datasets = ast.literal_eval(geoprep.parser.get("DATASETS", "datasets"))

    import multiprocessing
    num_cpus = max(1, int(multiprocessing.cpu_count() * geoprep.fraction_cpus))
    parallel_info = f"Yes ({num_cpus} CPUs)" if geoprep.parallel_process else "No"

    utils.display_run_summary("GeoDownload Runner", [
        ("Usage", "from geoprepare import geodownload; geodownload.run(cfg)"),
        ("cfg", "[geobase.txt]"),
        ("Datasets", datasets),
        ("Years", f"{geoprep.start_year} - {geoprep.end_year}"),
        ("Parallel", parallel_info),
        ("Download dir", str(geoprep.dir_download)),
        ("Intermed dir", str(geoprep.dir_intermed)),
    ])

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

            geoprep.start_year = geoprep.parser.getint("NSIDC", "start_year")
            geoprep.end_year = geoprep.parser.getint("NSIDC", "end_year")
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
        elif dataset == "AEF":
            from .datasets import AEF as obj

            # AEF configuration
            # Override start/end year (AEF data available 2018-2024 only)
            geoprep.start_year = geoprep.parser.getint("AEF", "start_year")
            geoprep.end_year = geoprep.parser.getint("AEF", "end_year")
            # Countries: read from geoextract.txt so the list is defined once
            from pathlib import Path
            config_dir = Path(path_config_file[0]).parent
            extract_parser = utils.read_config([str(config_dir / "geoextract.txt")])
            geoprep.aef_countries = ast.literal_eval(
                extract_parser.get("DEFAULT", "countries")
            )
            # Buffer in degrees around country extent
            geoprep.aef_buffer = geoprep.parser.getfloat(
                "AEF", "buffer", fallback=0.5
            )
            # Whether to download VRT files (needed to correct COG orientation)
            geoprep.aef_download_vrt = geoprep.parser.getboolean(
                "AEF", "download_vrt", fallback=True
            )
            # Optional path to cache the tile index
            geoprep.aef_index_cache = geoprep.parser.get(
                "AEF", "index_cache", fallback=None
            )
            # Data source: "gee" (Google Earth Engine) or "tiles" (source.coop HTTP)
            geoprep.aef_source = geoprep.parser.get(
                "AEF", "aef_source", fallback="gee"
            )
        elif dataset == "DAYMET":
            from .datasets import Daymet as obj

            # Daymet V4 daily weather (North America only).
            # Requires: pip install "geoprepare[daymet]" + NASA Earthdata Login.
            geoprep.daymet_bbox = ast.literal_eval(
                geoprep.parser.get("DAYMET", "bbox")
            )
            geoprep.daymet_variables = ast.literal_eval(
                geoprep.parser.get(
                    "DAYMET", "variables",
                    fallback="['tmin', 'tmax', 'prcp']",
                )
            )
            # Optional per-dataset year override (defaults to DEFAULT years)
            geoprep.start_year = geoprep.parser.getint(
                "DAYMET", "start_year", fallback=geoprep.start_year
            )
            geoprep.end_year = geoprep.parser.getint(
                "DAYMET", "end_year", fallback=geoprep.end_year
            )
        elif dataset == "CHIRTS-ERA5":
            from .datasets import CHIRTS_ERA5 as obj

            geoprep.fill_value = geoprep.parser.getint(
                "CHIRTS-ERA5", "fill_value", fallback=-9999
            )
            geoprep.chirts_variables = ast.literal_eval(
                geoprep.parser.get(
                    "CHIRTS-ERA5", "variables", fallback="['tmax', 'tmin']"
                )
            )
        elif dataset == "NOAA-S2S":
            from .datasets import NOAA_S2S as obj
        else:
            raise ValueError(f"{dataset} not implemented")

        # Print all elements of configuration file
        geoprep.pp_config(dataset)
        # Execute!
        obj.run(geoprep)

    print_download_summary(geoprep, datasets)


def _get_intermed_dirs(geoprep, dataset):
    """Return list of (label, directory) for a dataset's intermediate files."""
    d = geoprep.dir_intermed
    year = str(geoprep.end_year)

    if dataset == "CHIRPS":
        version = geoprep.parser.get("CHIRPS", "version", fallback="v2")
        return [("chirps", d / "chirps" / version / "global" / year)]
    elif dataset == "NDVI":
        return [("ndvi", d / "ndvi" / year)]
    elif dataset == "VIIRS":
        return [("gcvi", d / "gcvi" / year)]
    elif dataset == "CPC":
        return [
            ("cpc_tmax", d / "cpc_tmax" / year),
            ("cpc_tmin", d / "cpc_tmin" / year),
            ("cpc_precip", d / "cpc_precip" / year),
        ]
    elif dataset == "ESI":
        return [
            ("esi_4wk", d / "esi_4wk" / year),
            ("esi_12wk", d / "esi_12wk" / year),
        ]
    elif dataset == "LST":
        return [("lst", d / "lst" / year)]
    elif dataset == "FPAR":
        return [("fpar", d / "fpar" / year)]
    elif dataset == "NSIDC":
        return [
            ("nsidc_surface", d / "nsidc" / "daily" / "surface" / year),
            ("nsidc_rootzone", d / "nsidc" / "daily" / "rootzone" / year),
        ]
    elif dataset == "SOIL-MOISTURE":
        return [
            ("soil_moisture_as1", d / "soil_moisture_as1" / year),
            ("soil_moisture_as2", d / "soil_moisture_as2" / year),
        ]
    elif dataset == "CHIRPS-GEFS":
        return [("chirps_gefs", d / "chirps_gefs" / year)]
    elif dataset == "FLDAS":
        return [("fldas", d / "fldas" / "global" / year)]
    elif dataset == "AEF":
        return [("aef", d / "aef")]
    elif dataset == "DAYMET":
        daymet_vars = ast.literal_eval(
            geoprep.parser.get(
                "DAYMET", "variables",
                fallback="['tmin', 'tmax', 'prcp']",
            )
        )
        return [(f"daymet_{v}", d / f"daymet_{v}" / year) for v in daymet_vars]
    elif dataset == "CHIRTS-ERA5":
        chirts_vars = ast.literal_eval(
            geoprep.parser.get(
                "CHIRTS-ERA5", "variables", fallback="['tmax', 'tmin']"
            )
        )
        return [(f"chirts_era5_{v}", d / f"chirts_era5_{v}" / year) for v in chirts_vars]
    elif dataset == "NOAA-S2S":
        dl = geoprep.dir_download
        return [("noaa_s2s", Path(dl) / "noaa_s2s")]
    else:
        return []


def _extract_date_from_filename(filename):
    """Extract a date string from a TIF filename using common patterns."""
    name = filename

    # CHIRPS-GEFS: data.{year}.{mm}{dd}.tif
    m = re.search(r'data\.(\d{4})\.(\d{2})(\d{2})', name)
    if m:
        try:
            dt = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # FLDAS monthly: fldas_{var}_{yyyymm}_lead
    m = re.search(r'fldas_\w+_(\d{4})(\d{2})_lead', name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # NSIDC: {year}_{doy}
    m = re.search(r'(\d{4})_(\d{3})_(?:surface|rootzone)', name)
    if m:
        try:
            dt = datetime.datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%j").date()
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # NDVI/GCVI: {year}.{doy}
    m = re.search(r'(\d{4})\.(\d{3})\.c6', name)
    if m:
        try:
            dt = datetime.datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%j").date()
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Generic year+doy pattern (CHIRPS, CPC, ESI, LST, FPAR, Soil Moisture)
    m = re.search(r'(\d{4})(\d{3})', name)
    if m:
        try:
            dt = datetime.datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%j").date()
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return "?"


def print_download_summary(geoprep, datasets):
    """Print a rich table showing the last available date per dataset and save to text file."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Download Summary", show_lines=False)
    table.add_column("Dataset", style="bold cyan")
    table.add_column("Variable", style="white")
    table.add_column("Last Date", style="green")
    table.add_column("Files", justify="right", style="yellow")

    rows = []
    for dataset in datasets:
        dirs = _get_intermed_dirs(geoprep, dataset)
        for label, dir_path in dirs:
            if dataset == "AEF":
                if dir_path.exists():
                    years = sorted(
                        p.parent.name for p in dir_path.rglob("aef_*_*.tif")
                        if p.parent.name.isdigit()
                    )
                    if years:
                        rows.append((dataset, label, f"{years[0]}-{years[-1]}", str(len(set(years)))))
                    else:
                        rows.append((dataset, label, "-", "0"))
                else:
                    rows.append((dataset, label, "-", "0"))
                continue

            if not dir_path.exists():
                rows.append((dataset, label, "-", "0"))
                continue

            tifs = sorted(f.name for f in dir_path.iterdir() if f.suffix == ".tif")
            if not tifs:
                rows.append((dataset, label, "-", "0"))
                continue

            last_date = _extract_date_from_filename(tifs[-1])
            rows.append((dataset, label, last_date, str(len(tifs))))

    for row in rows:
        table.add_row(*row)

    console = Console()
    console.print(table)

    # Save as plain text file to dir_inputs (parent of dir_intermed)
    summary_path = Path(geoprep.dir_intermed).parent / "download_summary.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Download Summary ({timestamp})", "=" * 60]
    lines.append(f"{'Dataset':<15} {'Variable':<20} {'Last Date':<12} {'Files':>6}")
    lines.append("-" * 60)
    for ds, var, date, count in rows:
        lines.append(f"{ds:<15} {var:<20} {date:<12} {count:>6}")
    summary_path.write_text("\n".join(lines) + "\n")
    console.print(f"[dim]Summary saved to {summary_path}[/dim]")


if __name__ == "__main__":
    run()