import logging
import os

import torch
from yolov7.utils.date_time import date_string

from yolov7.utils.git import git_describe


def select_device(device: str) -> torch.device:
    """get device to train on

    Args:
        device (str, optional): string description of device: "cpu", "cuda:0", "cuda:1", ....

    Returns:
        torch.device: device class
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == "cpu"

    if cpu:
        # force torch.cuda.is_available() = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:  # non-cpu device requested
        # set environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        # check availability
        if not torch.cuda.is_available():
            msg = f"CUDA unavailable, invalid device: {device} requested"
            logging.fatal(msg)
            raise ValueError(msg)

    return torch.device(device)


def check_batch_size(device: torch.device, batch_size: int) -> int:
    """check that batch_size is compatible with device_count

    Args:
        device (torch.device): device to train on
        batch_size (int): num samples per model pass

    Raises:
        ValueError: if batchsize not multiple of GPU count

    Returns:
        int: batch_size
    """
    if device.type != "cuda":
        return batch_size

    n = torch.cuda.device_count()
    if (n > 1 and batch_size) and not (batch_size % n == 0):
        msg = f"batch-size {batch_size} not multiple of GPU count {n}"
        logging.fatal(msg)
        raise ValueError(msg)
    return batch_size
