import numpy as np
import numpy.ma as ma
import rasterio
from itertools import repeat
from multiprocessing import Pool
import logging

from .util import raster as raster

# from ..glimpse import envi


log = logging.getLogger(__name__)


SUPPRESS_ERRORS = True


class UnableToExtractStats(Exception):
    pass


def get_var(var: str, indicator_arr: np.ndarray) -> ma.MaskedArray:
    """
    Process data using masked arrays based on variable type.
    Preserves original data type where possible, using masks for invalid values.

    Args:
        var: String identifier for the variable type (e.g., 'ndvi', 'gcvi', etc.)
        indicator_arr: NumPy array containing the raw data values

    Returns:
        ma.MaskedArray: Masked and scaled data array

    Raises:
        ValueError: If var is not a recognized variable type
        TypeError: If indicator_arr is not a numpy array
    """
    if not isinstance(indicator_arr, (np.ndarray, ma.MaskedArray)):
        raise TypeError("indicator_arr must be a numpy array")

    # Dictionary mapping variable types to their masking and scaling functions
    processors = {
        "ndvi": lambda x: ma.masked_outside(x, 50, 250),
        "gcvi": lambda x: ma.masked_outside(x, 0, 200000) / 10000.0,
        "esi_4wk": lambda x: ma.masked_where(
            (x + 4.0) * 10.0 < 0.0,
            (x + 4.0) * 10.0
        ),
        "esi_12wk": lambda x: ma.masked_where(
            (x + 4.0) * 10.0 < 0.0,
            (x + 4.0) * 10.0
        ),
        "soil_moisture_as1": lambda x: ma.masked_where(
            (x < 0.0) | (x == 9999.0),
            x
        ),
        "soil_moisture_as2": lambda x: ma.masked_where(
            (x < 0.0) | (x == 9999.0),
            x
        ),
        "nsidc_surface": lambda x: ma.masked_where(
            (x < 0.0) | (x == 9999.0),
            x
        ),
        "nsidc_rootzone": lambda x: ma.masked_where(
            (x < 0.0) | (x == 9999.0),
            x
        ),
        "chirps": lambda x: ma.masked_less(x, 0.0) / 100.0,
        "chirps_gefs": lambda x: ma.masked_less(x, 0.0) / 100.0,
        "cpc_tmax": lambda x: ma.masked_less(x, -273.15),
        "cpc_tmin": lambda x: ma.masked_less(x, -273.15),
        "lst": lambda x: ma.masked_where(
            x * 0.02 - 273.15 < -123.15,
            x * 0.02 - 273.15
        )
    }

    if var in processors:
        return processors[var](indicator_arr)
    elif var in ["default", "other"]:
        return ma.masked_less(indicator_arr, 0.0)
    else:
        raise ValueError(f"Unrecognized variable type: {var}")


def _get_dtype_conversion(ds):
    """
    Get dtype conversion parameters for a rasterio dataset.
    
    Args:
        ds: rasterio.DatasetReader instance
        
    Returns:
        dict or None: Conversion parameters (nodata, scale, offset) or None if no conversion needed
    """
    if ds.driver == "ENVI":
        # Note: envi module would need to be imported if used
        # return envi.get_dtype_conversion(ds.name)
        log.warning("ENVI driver detected but envi module not available")
        return None
    elif ds.nodatavals and any(v is not None for v in ds.nodatavals):
        return dict(nodata=ds.nodatavals)
    return None


def geom_extract(
    geometry,
    variable,
    indicator,
    stats_out=("mean", "std", "min", "max", "sum", "counts"),
    afi=None,
    classification=None,
    afi_thresh=None,
    thresh_type=None,
):
    """
    Extracts the indicator statistics on input geometry using the AFI as weights.

    Global variable SUPPRESS_ERRORS controls if a custom error (UnableToExtractStats) should be raised when it's not
    possible to extract stats with given parameters. By default it is set to suppress errors and only report a warning.
    This setup is for the use case when the function is called directly and can handle an empty output.
    The opposite case, when the errors are raised is used when this function is called in a multiprocessing pool and
    it's necessary to link a proper error message with a geometry/unit identifier.

    Handles heterogeneous datasets by using the tbx_util.raster.get_common_bounds_and_shape function.

    :param geometry: GeoJSON-like feature (implements __geo_interface__) — feature collection, or geometry.
    :param variable: name of the variable to extract
    :param indicator: path to raster file or an already opened dataset (rasterio.DatasetReader) on which statistics are extracted
    :param stats_out: definition of statistics to extract, the list is directly forwarded to function
        asap_toolbox.util.raster.arr_stats.
        Additionally, accepts "counts" keyword that calculates following values:
            - total - overall unit grid coverage
            - valid_data - indicator without nodata
            - valid_data_after_masking - indicator used for calculation
            - weight_sum - total mask sum
            - weight_sum_used - mask sum after masking of dataset nodata is applied
    :param afi: path to Area Fraction index or weights - path to raster file or an already opened dataset (rasterio.DatasetReader)
    :param afi_thresh: threshold to mask out the afi data
    :param classification: If defined, calculates the pixel/weight sums of each class defined.
        Defined as JSON dictionary with borders as list of min, max value pairs and border behaviour definition:
            {
                borders: ((min1, max1), (min2, max2), ..., (min_n, max_n)),
                border_include: [min|max|both|None]
            }
    :return: dict with extracted stats divided in 3 groups:
        - stats - dict with calculated stats values (mean, std, min, max)
        - counts - dict with calculated count values (total; valid_data; valid_data_after_masking; weight_sum; weight_sum_used)
        - classification - dict with border definitions and values
        {
            stats: {mean: val, std: min: val, max: val, ...}
            counts: {total: val, valid_data: valid_data_after_masking: val, weight_sum: val, ...}
            classification: {
                borders: ((min1, max1), (min2, max2), ..., (min_n, max_n)),
                border_include: val,
                values: (val1, val2, val3,...)
            }
        }
        raises UnableToExtractStats error if geom outside raster, if the geometry didn't catch any pixels
    """
    output = dict()
    
    # Track whether we opened the datasets (so we know whether to close them)
    indicator_opened = False
    afi_opened = False
    
    try:
        # Make sure inputs are opened
        if isinstance(indicator, rasterio.DatasetReader):
            indicator_ds = indicator
        else:
            indicator_ds = rasterio.open(indicator)
            indicator_opened = True
            
        rasters_list = [indicator_ds]
        afi_ds = None
        
        if afi:
            if isinstance(afi, rasterio.DatasetReader):
                afi_ds = afi
            else:
                afi_ds = rasterio.open(afi)
                afi_opened = True
            rasters_list.append(afi_ds)

        # Get unified read window, bounds and resolution if heterogeneous resolutions
        try:
            read_bounds, read_shape = raster.get_common_bounds_and_shape(
                [geometry], rasters_list
            )
        except rasterio.errors.WindowError:
            e_msg = "Geometry has no intersection with the indicator"
            if SUPPRESS_ERRORS:
                log.warning("Skipping extraction! " + e_msg)
                return output
            else:
                raise UnableToExtractStats(e_msg)

        # Fetch indicator array
        indicator_arr = raster.read_masked(
            ds=indicator_ds,
            mask=[geometry],
            window=indicator_ds.window(*read_bounds),
            indexes=None,
            use_pixels="CENTER",
            out_shape=read_shape,
        )
        geom_mask = indicator_arr.mask
        indicator_arr = get_var(variable, indicator_arr)
        
        # Skip extraction if no pixels caught by geom
        if np.all(geom_mask):
            e_msg = "No pixels caught by geometry"
            if SUPPRESS_ERRORS:
                log.warning("Skipping extraction! " + e_msg)
                return output
            else:
                raise UnableToExtractStats(e_msg)
        
        # Convert pixel values if needed (ENVI file or has nodatavals)
        dtype_conversion = _get_dtype_conversion(indicator_ds)
        if dtype_conversion:
            indicator_arr = raster.arr_unpack(indicator_arr, **dtype_conversion)
        
        valid_data_mask = indicator_arr.mask

        # Fetch mask array
        afi_arr = None
        if afi and afi_ds:
            afi_arr = raster.read_masked(
                ds=afi_ds,
                mask=[geometry],
                indexes=None,
                window=afi_ds.window(*read_bounds),
                use_pixels="CENTER",
                out_shape=read_shape,
            )

            if afi_thresh is not None:
                if thresh_type == "Fixed":
                    afi_arr[
                        ~np.isnan(afi_arr) & (afi_arr <= afi_thresh) & ~afi_arr.mask
                    ] = 0

                elif thresh_type == "Percentile":
                    m_afi_arr = afi_arr[~np.isnan(afi_arr) & (afi_arr > 0) & ~afi_arr.mask]

                    if len(m_afi_arr) > 0:
                        thresh_PT = np.percentile(m_afi_arr, afi_thresh)

                        afi_arr[
                            ~np.isnan(afi_arr) & (afi_arr < thresh_PT) & ~afi_arr.mask
                        ] = 0

                afi_arr = np.ma.array(afi_arr, mask=(afi_arr.mask | (afi_arr == 0)))

            # Convert pixel values if needed for AFI
            afi_dtype_conversion = _get_dtype_conversion(afi_ds)
            if afi_dtype_conversion:
                afi_arr = raster.arr_unpack(afi_arr, **afi_dtype_conversion)
            
            # Apply the afi mask nodata mask to the dataset
            indicator_arr = np.ma.array(indicator_arr, mask=(afi_arr.mask | (afi_arr == 0)))

        # Check if any data left after applying all the masks
        if np.sum(~indicator_arr.mask) == 0:
            e_msg = "No data left after applying all the masks, mask sum == 0"
            if SUPPRESS_ERRORS:
                # log.warning('Skipping extraction! ' + e_msg)
                return output
            else:
                raise UnableToExtractStats(e_msg)

        # Extractions
        if any(val in ("min", "max", "mean", "sum", "std") for val in stats_out):
            output["stats"] = raster.arr_stats(
                indicator_arr, afi_arr if afi else None, stats_out
            )

        if "counts" in stats_out:
            output["counts"] = dict()
            # total - overall unit grid coverage
            output["counts"]["total"] = int((~geom_mask).sum())
            # valid_data - indicator without nodata
            output["counts"]["valid_data"] = int(np.sum(~valid_data_mask))
            if afi and afi_arr is not None:
                output["counts"]["valid_data_after_masking"] = int(
                    np.sum(~indicator_arr.mask)
                )
                # weight_sum - total mask sum
                output["counts"]["weight_sum"] = afi_arr.sum()
                if type(output["counts"]["weight_sum"]) == np.uint64:
                    output["counts"]["weight_sum"] = int(output["counts"]["weight_sum"])
                # weight_sum_used - mask sum after masking of dataset nodata is applied
                afi_arr_compressed = np.ma.array(
                    afi_arr, mask=indicator_arr.mask
                ).compressed()
                output["counts"]["weight_sum_used"] = afi_arr_compressed.sum()
                if type(output["counts"]["weight_sum_used"]) == np.uint64:
                    output["counts"]["weight_sum_used"] = int(
                        output["counts"]["weight_sum_used"]
                    )

        if classification:
            cls_def = [
                {"min": _min, "max": _max} for _min, _max in classification["borders"]
            ]
            classification_out = classification.copy()
            classification_out["border_include"] = classification.get(
                "border_include", "min"
            )
            class_res = raster.arr_classes_count(
                indicator_arr,
                cls_def=cls_def,
                weights=afi_arr if afi else None,
                border_include=classification_out["border_include"],
            )
            classification_out["values"] = [i["val_count"] for i in class_res]
            output["classification"] = classification_out

    finally:
        # Close datasets that we opened
        if indicator_opened and indicator_ds:
            indicator_ds.close()
        if afi_opened and afi_ds:
            afi_ds.close()

    return output