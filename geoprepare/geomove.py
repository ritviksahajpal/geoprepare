"""
geomove.py - Migrate existing flat dataset directories to year-specific subfolders.

Scans download and intermed directories for each dataset, extracts the year
from filenames via regex, and moves files into {base_dir}/{year}/ subfolders.

Pipeline: geomove (one-time migration) then geodownload/geoextract use year subfolders

Usage:
    from geoprepare import geomove
    geomove.run(["geobase.txt"], dry_run=True)   # preview
    geomove.run(["geobase.txt"])                  # execute
"""
import os
import re
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from . import base
from . import utils


# Each entry: (relative dir under download or intermed, regex with named group 'year', location)
# location is "download" or "intermed"
DATASET_PATTERNS = [
    # CPC
    ("cpc_tmax", r"cpc_(?P<year>\d{4})\d+_tmax_global\.tif", "intermed"),
    ("cpc_tmin", r"cpc_(?P<year>\d{4})\d+_tmin_global\.tif", "intermed"),
    ("cpc_precip", r"cpc_(?P<year>\d{4})\d+_precip_global\.tif", "intermed"),
    # ESI
    ("esi_4wk", r"esi_dfppm_4wk_(?P<year>\d{4})\d+\.tif", "intermed"),
    ("esi_12wk", r"esi_dfppm_12wk_(?P<year>\d{4})\d+\.tif", "intermed"),
    # NDVI
    ("ndvi", r"mod09\.ndvi\..*\.(?P<year>\d{4})\.\d{3}\..*\.tif", "intermed"),
    # NSIDC download
    ("nsidc", r"SMAP_.*_(?P<year>\d{4})\d{4}T\d+_.*\.h5", "download"),
    # NSIDC subdaily
    (os.path.join("nsidc", "subdaily"), r"nasa_usda_soil_moisture_(?P<year>\d{4})\d+T\d+_.*_global\.tif", "intermed"),
    # NSIDC daily surface
    (os.path.join("nsidc", "daily", "surface"), r"nasa_usda_soil_moisture_(?P<year>\d{4})_\d+_surface_global\.tif", "intermed"),
    # NSIDC daily rootzone
    (os.path.join("nsidc", "daily", "rootzone"), r"nasa_usda_soil_moisture_(?P<year>\d{4})_\d+_rootzone_global\.tif", "intermed"),
    # CHIRPS-GEFS
    ("chirps_gefs", r"data-mean_(?P<year>\d{4})\d+\.tif", "download"),
    ("chirps_gefs", r"data\.(?P<year>\d{4})\.\d+\.tif", "intermed"),
    # LST download
    ("modis_lst", r"MOD11C1\.A(?P<year>\d{4})\d+.*\.hdf", "download"),
    # LST intermed (old files were in dir_intermed root, new ones in lst/)
    ("lst", r"MOD11C1\.A(?P<year>\d{4})\d+_global\.tif", "intermed"),
    # Soil Moisture download
    (os.path.join("soil_moisture_nasa_usda", "grib"), r"(?P<year>\d{4})\d{4}.*\.grb2", "download"),
    # Soil Moisture intermed
    ("soil_moisture_as1", r"nasa_usda_soil_moisture_(?P<year>\d{4})\d+_as1_global\.tif", "intermed"),
    ("soil_moisture_as2", r"nasa_usda_soil_moisture_(?P<year>\d{4})\d+_as2_global\.tif", "intermed"),
    # AgERA5 intermed (per-variable subdirs)
    (os.path.join("agera5", "tif"), None, "intermed"),  # handled specially
    # VHI
    (os.path.join("vhi", "global"), r".*(?P<year>\d{4}).*VCI\.tif", "intermed"),
    # FPAR
    ("fpar", r"MCD15A2H\.A(?P<year>\d{4})\d+.*\.tif", "download"),
    # AEF intermed (per-country subdirs, skip avg files)
    ("aef", None, "intermed"),  # handled specially
]

# AgERA5 variable names for per-variable subdir scanning
AGERA5_VARS = [
    "Temperature_Air_2m_Mean_24h",
    "Temperature_Air_2m_Mean_Day_Time",
    "Temperature_Air_2m_Mean_Night_Time",
    "Dew_Point_Temperature_2m_Mean",
    "Temperature_Air_2m_Max_24h",
    "Temperature_Air_2m_Min_24h",
    "Temperature_Air_2m_Max_Day_Time",
    "Temperature_Air_2m_Min_Night_Time",
    "Precipitation_Flux",
    "Snow_Thickness_Mean",
    "Solar_Radiation_Flux",
    "Vapour_Pressure_Mean",
]


class GeoMove(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section="DEFAULT"):
        super().parse_config(section="DEFAULT")
        self.parallel_move = self.parser.getboolean(section, "parallel_move")


def _is_in_year_subdir(filepath):
    """Check if file is already in a year subfolder (parent dir name is 4 digits)."""
    return bool(re.fullmatch(r"\d{4}", filepath.parent.name))


def _move_files(base_dir, pattern, dry_run, logger):
    """
    Scan base_dir for files matching pattern, extract year, move to year subfolder.

    Returns:
        tuple: (moved, already, unmatched) counts
    """
    moved = 0
    already = 0
    unmatched = 0

    if not base_dir.exists():
        return moved, already, unmatched

    regex = re.compile(pattern)

    for f in sorted(base_dir.iterdir()):
        if not f.is_file():
            continue
        if _is_in_year_subdir(f):
            already += 1
            continue

        m = regex.match(f.name)
        if m:
            year = m.group("year")
            dest_dir = base_dir / year
            dest = dest_dir / f.name

            if dry_run:
                logger.info(f"[DRY RUN] Would move {f} -> {dest}")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
                logger.info(f"Moved {f.name} -> {year}/{f.name}")
            moved += 1
        else:
            unmatched += 1

    return moved, already, unmatched


def _move_agera5_files(dir_intermed, dry_run, logger):
    """Handle AgERA5 per-variable subdirectories."""
    total_moved = total_already = total_unmatched = 0
    agera5_pattern = r"agera5_(?P<year>\d{4})\d+_.*_global\.tif"

    agera5_tif_dir = dir_intermed / "agera5" / "tif"
    if not agera5_tif_dir.exists():
        return total_moved, total_already, total_unmatched

    for var_dir in sorted(agera5_tif_dir.iterdir()):
        if not var_dir.is_dir():
            continue
        m, a, u = _move_files(var_dir, agera5_pattern, dry_run, logger)
        total_moved += m
        total_already += a
        total_unmatched += u

    return total_moved, total_already, total_unmatched


def _move_aef_files(dir_intermed, dry_run, logger):
    """Handle AEF per-country subdirectories (skip avg files)."""
    total_moved = total_already = total_unmatched = 0
    aef_pattern = r"aef_(?P<year>\d{4})_\w+\.tif"

    aef_dir = dir_intermed / "aef"
    if not aef_dir.exists():
        return total_moved, total_already, total_unmatched

    for country_dir in sorted(aef_dir.iterdir()):
        if not country_dir.is_dir():
            continue
        m, a, u = _move_files(country_dir, aef_pattern, dry_run, logger)
        total_moved += m
        total_already += a
        total_unmatched += u

    return total_moved, total_already, total_unmatched


def _move_lst_from_root(dir_intermed, dry_run, logger):
    """Move old LST files from dir_intermed root to dir_intermed/lst/{year}/."""
    moved = 0
    pattern = re.compile(r"MOD11C1\.A(?P<year>\d{4})\d+_global\.tif")

    for f in sorted(dir_intermed.iterdir()):
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            year = m.group("year")
            dest_dir = dir_intermed / "lst" / year
            dest = dest_dir / f.name

            if dry_run:
                logger.info(f"[DRY RUN] Would move {f} -> {dest}")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
                logger.info(f"Moved {f.name} -> lst/{year}/{f.name}")
            moved += 1

    return moved


def _process_pattern(rel_dir, pattern, location, dir_download, dir_intermed, dry_run, logger):
    """Process a single dataset pattern. Returns (label, moved, already, unmatched)."""
    base_dir = dir_download / rel_dir if location == "download" else dir_intermed / rel_dir

    if not base_dir.exists():
        return rel_dir, 0, 0, 0

    m, a, u = _move_files(base_dir, pattern, dry_run, logger)
    return rel_dir, m, a, u


def run(path_config_file=["geobase.txt"], dry_run=False):
    geomove = GeoMove(path_config_file)
    geomove.parse_config("DEFAULT")

    logger = geomove.logger
    dir_download = geomove.dir_download
    dir_intermed = geomove.dir_intermed
    parallel_move = geomove.parallel_move

    tag = "[DRY RUN] " if dry_run else ""
    logger.info(f"{tag}Starting file migration to year-specific subfolders")

    num_cpus = max(1, int(multiprocessing.cpu_count() * geomove.fraction_cpus))
    parallel_info = f"Yes ({num_cpus} CPUs)" if parallel_move else "No"

    utils.display_run_summary("GeoMove Runner", [
        ("Usage", "from geoprepare import geomove; geomove.run(cfg, dry_run=True)"),
        ("cfg", "[geobase.txt]"),
        ("Dry run", str(dry_run)),
        ("Parallel", parallel_info),
        ("Download dir", str(dir_download)),
        ("Intermed dir", str(dir_intermed)),
    ])

    grand_moved = 0
    grand_already = 0
    grand_unmatched = 0

    # Handle old LST files in dir_intermed root
    lst_moved = _move_lst_from_root(dir_intermed, dry_run, logger)
    grand_moved += lst_moved
    if lst_moved:
        logger.info(f"LST (from root): {lst_moved} files moved")

    # Build task list: (label, callable) for standard patterns + special cases
    tasks = []
    for rel_dir, pattern, location in DATASET_PATTERNS:
        if pattern is None:
            continue
        tasks.append((rel_dir, pattern, location))

    if parallel_move:
        # -- PARALLEL EXECUTION --
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = {}
            for rel_dir, pattern, location in tasks:
                f = executor.submit(
                    _process_pattern, rel_dir, pattern, location,
                    dir_download, dir_intermed, dry_run, logger,
                )
                futures[f] = rel_dir
            # Special cases
            futures[executor.submit(_move_agera5_files, dir_intermed, dry_run, logger)] = "agera5"
            futures[executor.submit(_move_aef_files, dir_intermed, dry_run, logger)] = "aef"

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
                label = futures[future]
                result = future.result()
                if len(result) == 4:
                    _, m, a, u = result
                else:
                    m, a, u = result
                grand_moved += m
                grand_already += a
                grand_unmatched += u
                if m or a:
                    logger.info(f"{label}: {m} moved, {a} already in year folders, {u} unmatched")
    else:
        # -- SEQUENTIAL EXECUTION --
        for rel_dir, pattern, location in tqdm(tasks, desc="Processing datasets"):
            _, m, a, u = _process_pattern(
                rel_dir, pattern, location, dir_download, dir_intermed, dry_run, logger,
            )
            grand_moved += m
            grand_already += a
            grand_unmatched += u
            if m or a:
                logger.info(f"{rel_dir}: {m} moved, {a} already in year folders, {u} unmatched")

        # Handle AgERA5 special case
        m, a, u = _move_agera5_files(dir_intermed, dry_run, logger)
        grand_moved += m
        grand_already += a
        grand_unmatched += u
        if m or a:
            logger.info(f"agera5: {m} moved, {a} already in year folders, {u} unmatched")

        # Handle AEF special case
        m, a, u = _move_aef_files(dir_intermed, dry_run, logger)
        grand_moved += m
        grand_already += a
        grand_unmatched += u
        if m or a:
            logger.info(f"aef: {m} moved, {a} already in year folders, {u} unmatched")

    logger.info(f"\n{tag}Migration summary:")
    logger.info(f"  {grand_moved} files {'would be ' if dry_run else ''}moved")
    logger.info(f"  {grand_already} files already in year folders")
    logger.info(f"  {grand_unmatched} files unmatched (skipped)")


if __name__ == "__main__":
    run()
