import os
import re
import time
import gzip
import shutil
import logging
import functools
from datetime import datetime, timedelta
import requests
import subprocess
from multiprocessing import Pool

from tqdm import tqdm

import numpy as np

import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge

import rioxarray


def _download_merra_2(self, date, out_dir, **kwargs):
    merra2_urls = []
    for i in range(
        5
    ):  # we are collecting the requested date along with 4 previous days
        m_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=i)
        m_year = m_date.strftime("%Y")
        m_month = m_date.strftime("%m").zfill(2)
        m_day = m_date.strftime("%d").zfill(2)
        page_url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2SDNXSLV.5.12.4/{m_year}/{m_month}/"
        try:
            page_object = requests.get(page_url)
            page_text = page_object.text
            # regular expression that matches file name of desired date file
            ex = re.compile(f"MERRA2\S*{m_year}{m_month}{m_day}.nc4")

            # matches desired file name from web page
            m_filename = re.search(ex, page_text).group()

            m_url = page_url + m_filename
            merra2_urls.append(m_url)

        except AttributeError:
            return False

    # dictionary of empty lists of metric-specific file paths, waiting to be filled
    merra_datasets = {}
    merra_datasets["min"] = []
    merra_datasets["max"] = []
    merra_datasets["mean"] = []

    for url in merra2_urls:
        url_date = url.split(".")[-2]
        # use requests module to download MERRA-2 file (.nc4)
        # Athenticated using netrc method
        r = requests.get(url)
        out_netcdf = os.path.join(
            out_dir, f"merra-2.{url_date}.NETCDF.TEMP.nc"
        )  # NetCDF file path

        # write output .nc4 file
        with open(out_netcdf, "wb") as fd:  # write data in chunks
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

        # merra-2-max
        max_path = f"netcdf:{os.path.abspath(out_netcdf)}:T2MMAX"
        max_dataset = rasterio.open(max_path)
        merra_datasets["max"].append(max_dataset)

        # memerra-2-mean
        mean_path = f"netcdf:{os.path.abspath(out_netcdf)}:T2MMEAN"
        mean_dataset = rasterio.open(mean_path)
        merra_datasets["mean"].append(mean_dataset)

        # memerra-2-min
        min_path = f"netcdf:{os.path.abspath(out_netcdf)}:T2MMIN"
        min_dataset = rasterio.open(min_path)
        merra_datasets["min"].append(min_dataset)

        merra_out = []

        for metric in merra_datasets.keys():
            directory = os.path.join(out_dir, f"{self.product}-{metric}-temp")
            out = os.path.join(directory, f"merra-2-{metric}-temp.{date}.tif")

            dataset_list = merra_datasets[metric]
            mosaic, out_transform = merge(dataset_list)
            out_meta = dataset_list[0].meta.copy()

            crs = CRS.from_epsg(4326)
            out_meta.update({"driver": "GTiff"})
            out_meta.update({"crs": crs})

            with rasterio.open(out, "w", **out_meta) as dest:
                dest.write(mosaic)

            raster = rasterio.open(out)

            optimized = self._cloud_optimize(raster, out, nodata=False)
            if optimized:
                merra_out.append(out)

        os.remove(out_netcdf)

        return merra_out


def run(params):
    pass


if __name__ == "__main__":
    pass
