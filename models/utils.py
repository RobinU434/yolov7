from typing import List
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from torch.serialization import MAP_LOCATION
from yolov7.models.experimental import Ensemble
from yolov7.utils.google_utils import attempt_download
from yolov7.models.common import Conv


def load_model(
    weights: str | List[str],
    map_location: MAP_LOCATION = None,
) -> ModuleList | Module:
    """Loads an ensemble of models
    - weights=[a,b,c]
    - a single model with: weights=List[str] or weights=str

    Args:
        weights (str | List[str]): path to checkpoints
        map_location (MAP_LOCATION, optional): a function, torch.device, string or a dict specifying how to remap storage locations. Defaults to None.

    Returns:
        ModuleList| Module: A single module or an ensemble of Modules
    """
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # attempt_download(w)
        print(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        # FP32 model
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval())

    # Compatibility updates
    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif isinstance(m, Conv):
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
