import os
import tarfile
import requests
from tqdm import tqdm
import multiprocessing
from osgeo import gdal, osr
from osgeo.gdalnumeric import *
from bs4 import BeautifulSoup


def get_webpage_content(url):
    """
   Fetch and return the content of a webpage.
   Raises HTTPError for bad requests.

   Parameters:
   - url: The URL of the webpage to fetch.

   Returns:
   - The text content of the webpage.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX

    return response.text


def find_links(soup, file_type=".tar.gz"):
    """
    Find and return all .tar.gz links in the provided BeautifulSoup object.

    Parameters:
    - soup: BeautifulSoup object containing the parsed HTML of the page.

    Returns:
    - A list of .tar.gz file links found on the page.
    """
    return [a['href'] for a in soup.find_all('a') if a['href'].endswith(file_type)]


def download_and_extract_files(links, base_url, download_folder, interim_folder=None):
    """
    Download and extract each .tar.gz file found in the links.

    Parameters:
    - links: A list of .tar.gz file links to download.
    - base_url: The base URL to prepend to each link for downloading.
    - download_folder: The folder where files should be downloaded.
    - interim_folder: The folder where files should be extracted.
    """
    pbar = tqdm(links, total=len(links))
    for link in pbar:
        file_url = base_url + link
        file_path = download_folder / link

        if not os.path.exists(file_path):
            pbar.set_description(f"Downloading VHI file: {file_url}")
            pbar.update()

            file_response = requests.get(file_url)
            with open(file_path, 'wb') as file:
                file.write(file_response.content)

            if interim_folder:
                with tarfile.open(file_path, "r:gz") as tar:
                    tar.extractall(path=interim_folder)


def convert_to_global(params, input_vhi_file, output_global_file):

    # Then rematch tif file to correct resolution
    if not os.path.isfile(final_fl):
        xds = xr.open_dataarray(prelim_fl)
        xds_match = xr.open_dataarray(template_fl)

        xds_repr_match = xds.rio.reproject_match(xds_match)
        xds_repr_match.rio.to_raster(final_fl)

    # Define the spatial resolution (degrees per pixel) - for example, 1 degree.
    # This means each pixel represents 1 degree of latitude/longitude.
    x_res = 0.05
    y_res = 0.05

    # Calculate the dimensions based on the global extent and resolution
    x_size = int(360 / x_res)
    y_size = int(180 / y_res)
    breakpoint()
    # Create the raster dataset
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(input_vhi_file, x_size, y_size, 1, gdal.GDT_Float32)

    # Set the geo-transform (maps pixel coordinates to geographic coordinates)
    # GeoTransform parameters are: top left x, w-e pixel resolution, rotation (0 if North is up),
    # top left y, rotation (0 if North is up), and n-s pixel resolution (negative since coordinates decrease from North to South).
    dataset.SetGeoTransform([-180, x_res, 0, 90, 0, -y_res])

    # Define the spatial reference (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # EPSG for WGS84
    dataset.SetProjection(srs.ExportToWkt())

    # Get the raster band and fill it with zeros
    band = dataset.GetRasterBand(1)
    band.Fill(0)

    # Close the dataset to flush changes
    dataset = None


def process(all_params):
    """
    Main function to orchestrate the download of VHI data.

    Parameters:
    - all_params: A tuple containing the parameters and the year for which data is to be downloaded.
    """
    params = all_params[0]
    download_folder = params.dir_download / 'vhi'
    os.makedirs(download_folder, exist_ok=True)

    # Download historic data which is in the form of .tar.gz files
    response_text = get_webpage_content(params.url_historic)
    soup = BeautifulSoup(response_text, 'html.parser')
    tar_gz_links = find_links(soup, ".tar.gz")

    # Unzip the downloaded .tar.gz files
    interim_folder = params.dir_interim / "vhi" / "unzipped"
    os.makedirs(interim_folder, exist_ok=True)
    # download_and_extract_files(tar_gz_links, params.url_historic, download_folder, interim_folder)
    #
    # # Download the present data which is in the form of .tif files
    # response_text = get_webpage_content(params.url_current)
    # soup = BeautifulSoup(response_text, 'html.parser')
    # tif_links = find_links(soup, ".tif")
    #
    # download_and_extract_files(tif_links, params.url_current, interim_folder)

    # Convert all files to global data
    filelist = list(interim_folder.glob("*VCI.tif"))
    pbar = tqdm(filelist, total=len(filelist))
    for f in pbar:
        output_global_file = params.dir_interim / "vhi" / "global" / f.name
        pbar.set_description(f"Converting to global: {f.name}")
        pbar.update()

        if not os.path.isfile(output_global_file):
            convert_to_global(params, f, output_global_file)


def run(params):
    # Download VHI data
    if params.parallel_process:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            for i, _ in enumerate(p.imap_unordered(process, [params])):
                pass
    else:
        process([params])


if __name__ == "__main__":
    pass
