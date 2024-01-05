from torch import Tensor
from numpy import ndarray
import numpy as np


def xyxy2xywh(x: Tensor | ndarray) -> Tensor | ndarray:
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    where:
    - x1, y1 = top-left
    - x2, y2 = bottom-right

    Args:
        x (Tensor | ndarray) -> tensor to convert

    Returns:
        Tensor | ndarray: converted box
    """
    #
    y = x.clone() if isinstance(x, Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x: Tensor | ndarray) -> Tensor | ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    where:
    - x1, y1 = top-left
    - x2, y2 = bottom-right

    Args:
        x (Tensor | ndarray) -> tensor to convert

    Returns:
        Tensor | ndarray: converted box
    """
    y = x.clone() if isinstance(x, Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(
    x: Tensor | ndarray, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0
) -> Tensor | ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    where:
    - x1, y1 = top-left
    - x2, y2 = bottom-right

    Args:
        x (Tensor | ndarray) -> tensor to convert

    Returns:
        Tensor | ndarray: converted box
    """
    y = x.clone() if isinstance(x, Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(
    x: Tensor | ndarray, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0
) -> Tensor | ndarray:
    """Convert normalized 2D points into pixel segments,

    Args:
        x (Tensor | ndarray): shape (n,2)
        w (int, optional): image width. Defaults to 640.
        h (int, optional): image height. Defaults to 640.
        padw (int, optional): padding in width. Defaults to 0.
        padh (int, optional): padding in height. Defaults to 0.

    Returns:
        Tensor | ndarray: converted point
    """
    y = x.clone() if isinstance(x, Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(
    segment: Tensor | ndarray, width: int = 640, height: int = 640
) -> Tensor | ndarray:
    """Convert 1 segment label to 1 box label
    applying inside-image constraint

    Example:
    (xy1, xy2, ...) to (xyxy)


    Args:
        segment (Tensor | ndarray): _description_
        width (int, optional): _description_. Defaults to 640.
        height (int, optional): _description_. Defaults to 640.

    Returns:
        Tensor | ndarray: _description_
    """
    x, y = segment.T  # segment xy
    is_inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[is_inside]
    y = y[is_inside]

    # xyxy
    if any(x):
        return np.array([x.min(), y.min(), x.max(), y.max()])
    else:
        return np.zeros((1, 4))


def segments2boxes(segments: Tensor | ndarray) -> Tensor | ndarray:
    """Convert segment labels to box labels

    Example:
    (cls, xy1, xy2, ...) to (cls, xywh)

    Args:
        segments (Tensor | ndarray): _description_

    Returns:
        Tensor | ndarray: _description_
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh
