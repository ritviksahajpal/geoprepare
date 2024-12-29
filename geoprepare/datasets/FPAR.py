import itertools
import os

import requests
import multiprocessing
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def download_file(input):
    """
    Downloads a file from the given URL and saves it in the current directory.
    Shows a progress bar for the download.
    Args:
        input:

    Returns:

    """
    params, url = input
    local_filename = url.split('/')[-1]
    os.makedirs(params.dir_download / "fpar", exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=local_filename)

        with open(params.dir_download / "fpar" / local_filename, 'wb') as f:
            for chunk in r.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

    return local_filename


def get_file_urls(url):
    """
    Scrapes the given directory URL for .tif file links and returns their URLs.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful
    soup = BeautifulSoup(response.text, 'html.parser')

    return [url + a['href'] for a in soup.find_all('a') if a['href'].endswith('.tif')]


def download_FPAR(params):
    """

    Args:
        params:

    Returns:

    """
    all_params = []
    file_urls = get_file_urls(params.data_dir)
    for url in file_urls:
        all_params.extend(list(itertools.product([params], [url])))

    ncpus = int(multiprocessing.cpu_count() * params.fraction_cpus)
    # Download files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        # include params as an argument in download_file
        list(tqdm(executor.map(download_file, all_params), total=len(file_urls), desc="Downloading files"))


def run(params):
    """

    Args:
        params ():

    Returns:

    """
    download_FPAR(params)


if __name__ == "__main__":
    pass
