import os
import tarfile
import requests
from tqdm import tqdm
import multiprocessing
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
    for link in tqdm(links, desc="Downloading VHI data"):
        file_url = base_url + link
        file_path = download_folder / link

        if os.path.exists(file_path):
            continue

        file_response = requests.get(file_url)
        with open(file_path, 'wb') as file:
            file.write(file_response.content)

        if interim_folder:
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=interim_folder)


def download_VHI(all_params):
    """
    Main function to orchestrate the download of VHI data.

    Parameters:
    - all_params: A tuple containing the parameters and the year for which data is to be downloaded.
    """
    params, year = all_params
    download_folder = params.dir_download / 'vhi'
    os.makedirs(download_folder, exist_ok=True)

    # Download historic data which is in the form of .tar.gz files
    response_text = get_webpage_content(params.url_historic)
    soup = BeautifulSoup(response_text, 'html.parser')
    tar_gz_links = find_links(soup, ".tar.gz")

    # Unzip the downloaded .tar.gz files
    interim_folder = params.dir_interim / "vhi" / "unzipped"
    os.makedirs(interim_folder, exist_ok=True)
    download_and_extract_files(tar_gz_links, params.url_historic, download_folder, interim_folder)

    # Download the present data which is in the form of .tif files
    response_text = get_webpage_content(params.url_current)
    soup = BeautifulSoup(response_text, 'html.parser')
    tif_links = find_links(soup, ".tif")

    interim_folder = params.dir_interim / "vhi" / "unzipped"
    os.makedirs(interim_folder, exist_ok=True)
    download_and_extract_files(tif_links, params.url_current, interim_folder)


def run(params):
    import itertools

    all_params = []
    for year in range(params.start_year, params.end_year + 1):
        all_params.extend(list(itertools.product([params], [year])))

    # Download VHI data
    if params.parallel_process:
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as p:
            with tqdm(total=len(all_params), desc="Download VHI") as pbar:
                for i, _ in tqdm(enumerate(p.imap_unordered(download_VHI, all_params))):
                    pbar.update()
    else:
        for val in all_params:
            download_VHI(val)


if __name__ == "__main__":
    pass
