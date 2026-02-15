"""
geocheck.py - Validate TIF files in the intermed directory.

Scans the intermediate-processing directory for each configured dataset,
checks that every expected TIF file is present and non-empty, and
optionally verifies GDAL-readability.  Writes a timestamped report to
dir_logs/check/.

Pipeline utility — run after download (geodownload.py) or extract to
catch missing/corrupt files before they propagate downstream.
"""
import ast
import os
from calendar import isleap
from datetime import datetime
from pathlib import Path

from . import base
from . import utils


class GeoCheck(base.BaseGeo):
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def parse_config(self, section="DEFAULT"):
        super().parse_config(section="DEFAULT")


# ---------------------------------------------------------------------------
# Expected-file generators per dataset
# ---------------------------------------------------------------------------

def _expected_chirps(dir_intermed, start_year, end_year, parser):
    """Yield (path, label) for every expected CHIRPS global TIF."""
    version = parser.get("CHIRPS", "version", fallback="v2")
    version_str = "v2.0" if version == "v2" else "v3.0"

    for year in range(start_year, end_year + 1):
        ndays = 366 if isleap(year) else 365
        base_dir = dir_intermed / "chirps" / version / "global" / str(year)
        for jd in range(1, ndays + 1):
            fname = f"chirps_{version_str}_{year}{jd:03d}_global.tif"
            yield base_dir / fname, f"CHIRPS/{year}/{fname}"


def _expected_ndvi(dir_intermed, start_year, end_year, parser):
    """Yield (path, label) for every expected NDVI TIF (8-day composites)."""
    product = parser.get("NDVI", "product", fallback="MOD09CMG")
    vi = parser.get("NDVI", "vi", fallback="ndvi")
    prefix = product[:5].lower()

    for year in range(start_year, end_year + 1):
        ndays = 366 if isleap(year) else 365
        base_dir = dir_intermed / "ndvi"
        for doy in range(1, ndays + 1, 8):
            fname = f"{prefix}.{vi}.global_0.05_degree.{year}.{doy:03d}.c6.v1.tif"
            yield base_dir / fname, f"NDVI/{fname}"


def _expected_esi(dir_intermed, start_year, end_year, parser):
    """Yield (path, label) for every expected ESI TIF (7-day intervals)."""
    products = ast.literal_eval(parser.get("ESI", "list_products", fallback="['4wk']"))

    for product in products:
        base_dir = dir_intermed / f"esi_{product}"
        for year in range(start_year, end_year + 1):
            ndays = 366 if isleap(year) else 365
            for jd in range(1, ndays + 1, 7):
                fname = f"esi_dfppm_{product}_{year}{jd:03d}.tif"
                yield base_dir / fname, f"ESI_{product}/{fname}"


def _expected_cpc(dir_intermed, start_year, end_year, parser):
    """Yield (path, label) for every expected CPC TIF."""
    for var in ("tmax", "tmin", "precip"):
        base_dir = dir_intermed / f"cpc_{var}"
        for year in range(start_year, end_year + 1):
            ndays = 366 if isleap(year) else 365
            for jd in range(1, ndays + 1):
                fname = f"cpc_{year}{jd:03d}_{var}_global.tif"
                yield base_dir / fname, f"CPC_{var}/{fname}"


def _expected_lst(dir_intermed, start_year, end_year, parser):
    """Yield (path, label) for every expected LST TIF."""
    for year in range(start_year, end_year + 1):
        ndays = 366 if isleap(year) else 365
        for jd in range(1, ndays + 1):
            fname = f"MOD11C1.A{year}{jd:03d}_global.tif"
            yield dir_intermed / fname, f"LST/{fname}"


def _expected_nsidc(dir_intermed, start_year, end_year, parser):
    """Yield (path, label) for every expected NSIDC daily TIF."""
    for var in ("surface", "rootzone"):
        base_dir = dir_intermed / "nsidc" / "daily" / var
        for year in range(start_year, end_year + 1):
            ndays = 366 if isleap(year) else 365
            for jd in range(1, ndays + 1):
                fname = f"nasa_usda_soil_moisture_{year}_{jd:03d}_{var}_global.tif"
                yield base_dir / fname, f"NSIDC_{var}/{fname}"


# Registry: dataset name -> generator function
_DATASET_GENERATORS = {
    "CHIRPS": _expected_chirps,
    "NDVI": _expected_ndvi,
    "VIIRS": _expected_ndvi,
    "ESI": _expected_esi,
    "CPC": _expected_cpc,
    "LST": _expected_lst,
    "NSIDC": _expected_nsidc,
}


# ---------------------------------------------------------------------------
# Core check logic
# ---------------------------------------------------------------------------

def _check_dataset(dataset, dir_intermed, start_year, end_year, parser, gdal_check=False):
    """Check a single dataset.  Returns (expected, missing, corrupt) counts and detail lists."""
    gen = _DATASET_GENERATORS.get(dataset)
    if gen is None:
        return None  # dataset not supported for checking

    missing = []
    corrupt = []
    expected = 0

    if gdal_check:
        from osgeo import gdal

    for path, label in gen(dir_intermed, start_year, end_year, parser):
        expected += 1
        if not path.exists() or path.stat().st_size == 0:
            missing.append(label)
            continue
        if gdal_check:
            ds = gdal.Open(str(path))
            if ds is None:
                corrupt.append(label)

    return expected, missing, corrupt


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(report_dir, datasets_results, start_year, end_year):
    """Write a timestamped plain-text report and return its path."""
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"check_{stamp}.txt"

    total_expected = 0
    total_missing = 0
    total_corrupt = 0

    lines = []
    lines.append(f"GeoCheck Report  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Year range: {start_year} – {end_year}")
    lines.append("")

    for dataset, result in datasets_results.items():
        if result is None:
            lines.append(f"[{dataset}]  — skipped (no checker available)")
            lines.append("")
            continue

        expected, missing, corrupt = result
        ok = expected - len(missing) - len(corrupt)
        total_expected += expected
        total_missing += len(missing)
        total_corrupt += len(corrupt)

        lines.append(f"[{dataset}]  expected={expected}  ok={ok}  missing={len(missing)}  corrupt={len(corrupt)}")

        if missing:
            lines.append(f"  Missing ({len(missing)}):")
            for m in missing[:50]:
                lines.append(f"    {m}")
            if len(missing) > 50:
                lines.append(f"    ... and {len(missing) - 50} more")

        if corrupt:
            lines.append(f"  Corrupt ({len(corrupt)}):")
            for c in corrupt[:50]:
                lines.append(f"    {c}")
            if len(corrupt) > 50:
                lines.append(f"    ... and {len(corrupt) - 50} more")

        lines.append("")

    # Summary table
    total_ok = total_expected - total_missing - total_corrupt
    lines.append("=" * 60)
    lines.append(f"TOTAL  expected={total_expected}  ok={total_ok}  missing={total_missing}  corrupt={total_corrupt}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    report_path.write_text(text)
    return report_path, text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(path_config_file=["geobase.txt"], gdal_check=False):
    obj = GeoCheck(path_config_file)
    obj.parse_config("DEFAULT")
    datasets = ast.literal_eval(obj.parser.get("DATASETS", "datasets"))

    utils.display_run_summary("GeoCheck Runner", [
        ("Datasets", datasets),
        ("Years", f"{obj.start_year} - {obj.end_year}"),
        ("Intermed dir", str(obj.dir_intermed)),
        ("GDAL check", str(gdal_check)),
    ])

    from tqdm import tqdm

    results = {}
    for dataset in tqdm(datasets, desc="Checking datasets"):
        results[dataset] = _check_dataset(
            dataset, obj.dir_intermed, obj.start_year, obj.end_year, obj.parser, gdal_check
        )

    report_dir = obj.dir_logs / "check"
    report_path, report_text = _write_report(report_dir, results, obj.start_year, obj.end_year)

    print(f"\n{report_text}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run()
