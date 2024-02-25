########################################################################################################################
# Ritvik Sahajpal, Joanne Hall
# ritvik@umd.edu
#
# The original data is in zipped/unzipped tiff format at 0.05 degree resolution (zipped for final products and unzipped
# for preliminary product in recent years). The naming convention is in year, month, day and to match the other datasets
# the data has to be renamed into year, julian day.
#
# Final Tiffs
# Step 1: Unzip, rename the original tiff files, and convert the floating point (unit: mm) data into integer by scaling
# by 100. This speeds up step 2 and the Weighted Average Extraction code.
# Step 2: Convert the data into a global extent to match the crop masks.
#
# Preliminary Tiffs
# Step 1: Rename the original tiff files. Convert the floating point (unit: mm) data into integer by scaling by 100.
# This speeds up step 2 and the Weighted Average Extraction code.
# Step 2: Convert the data into a global extent to match the crop masks.
########################################################################################################################
import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def download_FPAR(params):
    base_url = "https://agricultural-production-hotspots.ec.europa.eu/data/indicators_fpar/fpar/"

    dir_out = params.dir_download / "fpar"
    os.makedirs(dir_out, exist_ok=True)
    print(f"Downloading to {dir_out}")

    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in tqdm(
        soup.select("a[href$='.tif']"), desc=f"FPAR"
    ):
        # Name the pdf files using the last portion of each link which are unique in this case
        filename = os.path.join(dir_out, link["href"].split("/")[-1])

        if not os.path.isfile(filename):
            with open(filename, "wb") as f:
                f.write(requests.get(urljoin(params.data_url + "/", link["href"])).content)


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    ##########
    # DOWNLOAD
    ##########
    download_FPAR(params)


if __name__ == "__main__":
    pass
