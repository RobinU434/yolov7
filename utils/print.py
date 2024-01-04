import logging
import platform

import torch
from yolov7.utils.date_time import date_string
from yolov7.utils.git import git_describe


def print_status(device: torch.device) -> None:
    """print project status

    Args:
        device (torch.device): device
    """
    s = f"YOLOR 🚀 \n {git_describe() or date_string()} torch {torch.__version__} "

    if device.type == "cuda" and torch.cuda.is_available():
        space = " " * len(s)
        n = torch.cuda.device_count()
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB

    # emoji-safe
    if platform.system() == "Windows":
        s = s.encode().decode("ascii", "ignore")

    logging.info(s)
