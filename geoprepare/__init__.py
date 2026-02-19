"""Top-level package for geoprepare."""

__author__ = """Ritvik Sahajpal"""
__email__ = "ritvik@umd.edu"
__version__ = "0.6.107"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geoprepare")
except PackageNotFoundError:
    # Package is not installed, use hardcoded version
    pass

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    pass

__all__ = ["log", "utils", "base", "geodownload", "geoextract", "geomerge", "geocheck", "diagnostics"]