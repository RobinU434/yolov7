import os
import zipfile
from typing import List

import torch
from yolov7.utils.logging import info_log


def load_txt(path: str) -> List[str]:
    """opens and reads lines from txt file

    Args:
        path (str): path to txt file

    Returns:
        List[str]: content of txt file
    """
    with open(path, "r", encoding="utf-8") as file:
        content = file.readlines()
    content = [line.rstrip("\n").strip(" ") for line in content]
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
        info_log(f"Download: {url} -> {path}")
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
        if not check_existing_directory(directory):
            create_directory(directory)

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(directory)


def check_existing_directory(path: str) -> bool:
    """check if a given directory exists

    Args:
        path (str): path to directory

    Returns:
        bool: if directory exists
    """
    return os.path.isdir(path)


def create_directory(path: str):
    """creates the path to a given directory

    Args:
        path (str): path to directory
    """
    os.makedirs(path)
