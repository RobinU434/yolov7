import glob
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from yolov7.utils.logging import info_log


def load_txt(path: str, encoding: str = "utf-8") -> List[str]:
    """opens and reads lines from txt file

    Args:
        path (str): path to txt file
        encoding (str): encoding of txt file. Default set to "utf-8"

    Returns:
        List[str]: content of txt file
    """
    with open(path, "r", encoding=encoding) as file:
        content = file.readlines()
    content = [line.rstrip("\n").strip(" ") for line in content]
    return content


def load_yaml(path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """opens and reads a yaml file and returns the content dict

    Args:
        path (str): path to yaml file
        encoding (str): encoding of txt file. Default set to "utf-8"

    Returns:
        Dict[str, Any]: content of yaml file
    """
    with open(path, "r", encoding=encoding) as file:
        content = yaml.safe_load(file)
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


def increment_path(path: str | Path, exist_ok=True, sep="") -> Path:
    """Increment path

    Example:
    >>> increment_path("runs/exp")
    runs/exp{sep}0

    Args:
        path (str | Path): where to increment the path
        exist_ok (bool, optional): check if path already exists. Defaults to True.
        sep (str, optional): separator before index. Defaults to "".

    Returns:
        Path: incremented path
    """
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    if not isinstance(path, Path):
        path = Path(path)  # os-agnostic

    if (path.exists() and exist_ok) or (not path.exists()):
        path = str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = f"{path}{sep}{n}"  # update path

    return Path(path)
