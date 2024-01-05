import sys
import time
import warnings
from typing import List

import coremltools as ct
import onnx
import onnxsim
import torch
import torch.nn as nn
import yolov7.models as models
from torch import Tensor
from torch.nn import Module
from torch.utils.mobile_optimizer import optimize_for_mobile
from yolov7.dataset.utils.checks import check_img_size
from yolov7.models.experimental import End2End
from yolov7.models.utils.activations import Hardswish, SiLU
from yolov7.models.utils.load import load_model
from yolov7.utils.add_nms import RegisterNMS
from yolov7.utils.torch import select_device, set_map_location


class Exporter:
    def __init__(
        self,
        image_size: List[int] = [640, 640],
        batch_size: int = 1,
        grid: bool = False,
        end2end: bool = False,
        simplify: bool = False,
        include_nms: bool = False,
        int8: bool = False,
        float16: bool = False,
        dynamic: bool = False,
        dynamic_batch: bool = False,
        max_wh: int = None,
        topk_all: int = 100,
        iou_threshold: float = 0.45,
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self._image_size = image_size
        self._batch_size = batch_size

        self._grid = grid
        self._end2end = end2end
        self._simplify = simplify
        self._include_nms = include_nms
        self._int8 = int8
        self._float16 = float16
        self._dynamic = dynamic
        self._dynamic_batch = dynamic_batch

        self._max_wh = max_wh
        self._topk_all = topk_all
        self._iou_threshold = iou_threshold
        self._conf_threshold = conf_threshold
        self._device = select_device(device)

    def _checks(self, model: Module):
        grid_size = int(max(model.stride))  # grid size (max stride)
        # verify img_size are gs-multiples
        self._img_size = [check_img_size(x, grid_size) for x in self._image_size]

    def _sample_input(self, device) -> Tensor:
        # image size(1,3,320,192) iDetection
        img = torch.zeros(self._batch_size, 3, *self._image_size).to(device)
        print(img.size())
        return img

    def _update_model(self, model: Module):
        for _, m in model.named_modules():
            # pytorch 1.6.0 compatibility
            m._non_persistent_buffers_set = set()

            # assign export-friendly activations
            if isinstance(m, models.common.Conv):
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()

        model.model[-1].export = not self._grid  # set Detect() layer grid export
        # dry run
        model(self._sample_input(self._device))

        if self._include_nms:
            model.model[-1].include_nms = True

        return model

    def _export_torch_script(self, model: Module, path: str):
        try:
            print("\nStarting TorchScript export with torch %s..." % torch.__version__)
            ts = torch.jit.trace(model, self._sample_input(self._device), strict=False)
            ts.save(path)
            print("TorchScript export success, saved as %s" % path)
        except Exception as e:
            print("TorchScript export failure: %s" % e)

    def _export_CoreML(self, model: Module, path: str):
        try:
            print("\nStarting CoreML export with coremltools %s..." % ct.__version__)
            # convert model from torchscript and apply pixel scaling as per detect.py
            sample_input = self._sample_input(self._device)
            ts = torch.jit.trace(model, sample_input, strict=False)
            ct_model = ct.convert(
                ts,
                inputs=[
                    ct.ImageType(
                        "image",
                        shape=sample_input.shape,
                        scale=1 / 255.0,
                        bias=[0, 0, 0],
                    )
                ],
            )

            if self._int8:
                bits = 8
                mode = "kmeans_lut"
            elif self._float16:
                bits = 16
                mode = "linear"
            else:
                bits = 32
                mode = None

            # quantization only supported on macOS
            if bits < 32 and sys.platform.lower() == "darwin":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=DeprecationWarning
                    )  # suppress numpy==1.20 float warning
                    ct_model = (
                        ct.models.neural_network.quantization_utils.quantize_weights(
                            ct_model, bits, mode
                        )
                    )
            else:
                print("quantization only supported on macOS, skipping...")

            ct_model.save(path)
            print("CoreML export success, saved as %s" % path)
        except Exception as e:
            print("CoreML export failure: %s" % e)

    def _export_torch_script_lite(self, model: Module, path: str):
        try:
            print(
                "\nStarting TorchScript-Lite export with torch %s..."
                % torch.__version__
            )
            sample_input = self._sample_input(self._device)
            tsl = torch.jit.trace(model, sample_input, strict=False)
            tsl = optimize_for_mobile(tsl)
            tsl._save_for_lite_interpreter(path)
            print("TorchScript-Lite export success, saved as %s" % path)
        except Exception as e:
            print("TorchScript-Lite export failure: %s" % e)

    def _export_onnx(self, model: Module, path: str):
        # ONNX export
        try:
            print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
            if self._include_nms:
                output_names = ["classes", "boxes"]
            else:
                output_names = ["output"]
            model.eval()
            dynamic_axes = None
            if self._dynamic:
                dynamic_axes = {
                    "images": {
                        0: "batch",
                        2: "height",
                        3: "width",
                    },  # size(1,3,640,640)
                    "output": {0: "batch", 2: "y", 3: "x"},
                }
            if self._dynamic_batch:
                self._batch_size = "batch"
                dynamic_axes = {
                    "images": {
                        0: "batch",
                    },
                }
                if self._end2end and self._max_wh is None:
                    output_axes = {
                        "num_dets": {0: "batch"},
                        "det_boxes": {0: "batch"},
                        "det_scores": {0: "batch"},
                        "det_classes": {0: "batch"},
                    }
                else:
                    output_axes = {
                        "output": {0: "batch"},
                    }
                dynamic_axes.update(output_axes)
            if self._grid:
                if self._end2end:
                    print(
                        "\nStarting export end2end onnx model for %s..." % "TensorRT"
                        if self._max_wh is None
                        else "onnxruntime"
                    )
                    model = End2End(
                        model,
                        self._topk_all,
                        self._iou_threshold,
                        self._conf_threshold,
                        self._max_wh,
                        self._device,
                        len(model.names),
                    )
                    if self._max_wh is None:
                        output_names = [
                            "num_dets",
                            "det_boxes",
                            "det_scores",
                            "det_classes",
                        ]
                        shapes = [
                            self._batch_size,
                            1,
                            self._batch_size,
                            self._topk_all,
                            4,
                            self._batch_size,
                            self._topk_all,
                            self._batch_size,
                            self._topk_all,
                        ]
                    else:
                        output_names = ["output"]
                else:
                    model.model[-1].concat = True

            torch.onnx.export(
                model,
                self._sample_input(self._device),
                path,
                verbose=False,
                opset_version=12,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            # Checks
            onnx_model = onnx.load(path)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            if self._end2end and self._max_wh is None:
                for i in onnx_model.graph.output:
                    for j in i.type.tensor_type.shape.dim:
                        j.dim_param = str(shapes.pop(0))

            if self._simplify:
                try:
                    print("\nStarting to simplify ONNX...")
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, "assert check failed"
                except Exception as e:
                    print(f"Simplifier failure: {e}")

            # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
            onnx.save(onnx_model, path)
            print("ONNX export success, saved as %s" % path)

            if self._include_nms:
                print("Registering NMS plugin for ONNX...")
                mo = RegisterNMS(path)
                mo.register_nms()
                mo.save(path)

        except Exception as e:
            print("ONNX export failure: %s" % e)

    def export(self, model_path: str):
        start = time.perf_counter()
        device = select_device("cpu")
        map_location = set_map_location(device)
        model = load_model(model_path, map_location=map_location)

        self._checks(model)
        model = self._update_model(model)

        # TorchScript export
        path = f = model_path.replace(".pt", ".torchscript.pt")
        self._export_torch_script(model, path)

        # CoreML export
        path = model_path.replace(".pt", ".mlmodel")
        self._export_CoreML(model, path)

        # TorchScript-Lite export
        path = model_path.replace(".pt", ".torchscript.ptl")
        self._export_torch_script_lite(model, path)

        # ONNX export
        path = model_path.replace(".pt", ".onnx")  # filename
        self._export_onnx(model, path)

        # Finish
        print(
            "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
            % (time.perf_counter() - start)
        )
