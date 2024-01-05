import logging
import os

from yolov7.utils.arithmetic import make_divisible


def check_dataset_coco(path_to_dataset: str) -> bool:
    """checks if the coco dataset is existing in the expected directory structure

    Args:
        path_to_dataset (str): path to dataset parent directory

    Returns:
        bool: if the dataset exists as expected
    """
    required_files = set(
        [
            "train2017.txt",
            "val2017.txt",
            "instances_val2017.json",
        ]
    )
    required_directories = set(
        [
            "annotations",
            "labels",
            "images",
            "train2017",
            "val2017",
            "test2017",
        ]
    )

    existing_files = []
    existing_subdirs = []

    for _, dirs, files in os.walk(path_to_dataset):
        existing_files.extend(files)
        existing_subdirs.extend(dirs)

    existing_subdirs = set(existing_subdirs)
    existing_files = set(existing_files)

    # Check if all required subdirectories are present
    not_existing_files = required_files - existing_files
    not_existing_dirs = required_directories - existing_subdirs

    if len(not_existing_files) > 0:
        msg = f"Files: {not_existing_files} do not exist in {path_to_dataset}"
        logging.warning(msg)
        return False

    if len(not_existing_dirs) > 0:
        msg = f"Directories: {not_existing_dirs} do not exist in {path_to_dataset}"
        logging.warning(msg)
        return False

    return True


def check_img_size(img_size: int, s: int = 32) -> int:
    """Verify img_size is a multiple of stride s

    Args:
        img_size (int): _description_
        s (int, optional): _description_. Defaults to 32.

    Returns:
        int: _description_
    """
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size
