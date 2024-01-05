import logging
import os
import time

import torch
from torch.serialization import MAP_LOCATION


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


def set_map_location(device: torch.device) -> MAP_LOCATION:
    """returns in case a map to map the storage location from cuda to cpu

    Args:
        device (torch.device): device

    Returns:
        MAP_LOCATION: is either None or a torch.device
    """
    if device.type == "cpu":
        return device
    return None


def time_synchronized() -> float:
    """get pytorch-accurate time

    Returns:
        float: accurate time
    """
    #
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
