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
    s = f"YOLOR 🚀 \n {git_describe() or date_string()} \n torch: {torch.__version__} "

    if device.type == "cuda" and torch.cuda.is_available():
        space = " " * len(s)
        n = torch.cuda.device_count()
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB

    # emoji-safe
    if platform.system() == "Windows":
        s = s.encode().decode("ascii", "ignore")

    print(s)


def colorstr(*input: str) -> str:
    """Colors a string
    https://en.wikipedia.org/wiki/ANSI_escape_code,

    Example:
    >>> colorstr('blue', 'hello world')


    Returns:
        str: colored string
    """
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
