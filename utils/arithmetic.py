import math


def make_divisible(x: float, divisor: float) -> float:
    """Returns x evenly divisible by divisor

    Args:
        x (float): value to adapt
        divisor (float):

    Returns:
        float: x divisible by divisor
    """
    return math.ceil(x / divisor) * divisor
