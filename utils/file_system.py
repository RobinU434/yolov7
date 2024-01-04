import zipfile
from typing import List

import torch


def load_txt(path: str) -> List[str]:
    """opens and reads lines from txt file

    Args:
        path (str): path to txt file

    Returns:
        List[str]: content of txt file
    """
    with open(path, "r", encoding="utf-8") as file:
        content = file.readlines()
    return content


def download_zip(urls: List[str], destination: str) -> List[str]:
    """download a list of zip files from the internet

    Args:
        urls (List[str]): list of urls to zip files
        destination (str): where to store the downloaded zips

    Returns:
        List[str]: file system paths to the zip files
    """
    paths = []
    for url in urls:
        file_name = url.split("/")[-1]
        path = destination + "/" + file_name
        torch.hub.download_url_to_file(url, path)
        paths.append(path)
    return paths


def unpack_zip(paths: List[str]):
    """unpack given list of zip files

    Args:
        paths (List[str]): paths to zip files
    """
    for path in paths:
        directory = path.rstrip(".zip")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(directory)
