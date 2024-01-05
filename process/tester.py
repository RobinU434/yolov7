from pathlib import Path
from threading import Thread
from typing import List
import torch
from torch import Tensor
from tqdm import tqdm
from yolov7.criterion.metrics import non_max_suppression
from yolov7.dataset.utils.load import load_coco_dataloader
from yolov7.models.utils.convert import output_to_target
from yolov7.models.utils.load import load_model
from yolov7.utils.file_system import increment_path
from yolov7.utils.plot.output import plot_images
from yolov7.utils.print import print_status
from yolov7.utils.torch import (
    check_batch_size,
    select_device,
    set_map_location,
)


class Tester:
    def __init__(
        self,
        model_path: str,
        data_path: str = "yolov7",
        batch_size: int = 32,
        device: str = "cpu",
        save_dir: str = "runs/test/exp",
        half_precision: bool = True,
        conf_threshold: float = 1e-3,
        iou_threshold: float = 0.6,
        plot: bool = False,
    ) -> None:
        self._device = select_device(device)
        self._model = load_model(model_path, set_map_location(self._device))

        self._half_precision = half_precision
        if self._device.type != "cpu" and self._half_precision:
            self._model.half()
        else:
            self._half_precision = False

        self._model.eval()

        self._batch_size = check_batch_size(self._device, batch_size)
        grid_stride = max(int(self._model.stride.max()), 32)
        self._test_dataloader = load_coco_dataloader(
            data_path,
            entity="test",
            batch_size=self._batch_size,
            grid_stride=grid_stride,
        )

        self._save_dir = increment_path(save_dir, exist_ok=False)
        self._save_dir.mkdir(parents=True, exist_ok=False)

        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._plot = plot

        print_status(self._device)

    def _single_pass(
        self, batch_idx: int, img: Tensor, targets: Tensor, paths: List[str], shapes
    ):
        img = img.to(self._device, non_blocking=True)
        # uint8 to fp16/32
        if self._half_precision:
            img = img.half()
        else:
            img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        targets = targets.to(self._device)

        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            # t = time_synchronized()
            out, train_out = self._model(
                img, augment=False
            )  # inference and training outputs
            # t0 += time_synchronized() - t

            """# Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][
                    :3
                ]  # box, obj, cls"""

            # Run NMS
            # to pixels
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
                self._device
            )
            # for autolabelling
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            # t = time_synchronized()
            out = non_max_suppression(
                out,
                conf_threshold=self._conf_threshold,
                iou_threshold=self._iou_threshold,
                labels=lb,
                multi_label=True,
            )
            # t1 += time_synchronized() - t

        # Statistics per image
        # self._do_statistics(out, targets, paths)

        # Plot images
        if self._plot and batch_idx < 3:
            self._plot_results(
                batch_idx=batch_idx, img=img, out=out, targets=targets, paths=paths
            )

    def _do_statistics(self, out: Tensor, targets: Tensor, paths: List[str]):
        stats = []
        iou = torch.linspace(0.5, 0.95, 10)
        for out_idx, pred in enumerate(out):
            labels = targets[targets[:, 0] == out_idx, 1:]
            num_labels = len(labels)
            target_class = labels[:, 0].tolist() if num_labels else []
            path = Path(paths[out_idx])

            # no bounding boxes predicted
            if len(pred) == 0:
                if num_labels:
                    stats.append(
                        (
                            torch.zeros(0, len(iou), dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            target_class,
                        )
                    )
                continue

            # Predictions
            """predn = pred.clone()
            scale_coords(
                img[out_idx].shape[1:], predn[:, :4], shapes[out_idx][0], shapes[out_idx][1]
            )  # native-space pred"""

            # Append to text file
            """if save_txt:
                gn = torch.tensor(shapes[out_idx][0])[
                    [1, 0, 1, 0]
                ]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    line = (
                        (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    )  # label format
                    with open(save_dir / "labels" / (path.stem + ".txt"), "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")
            """

            # W&B logging - Media Panel Plots
            """if (
                len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0
            ):  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [
                        {
                            "position": {
                                "minX": xyxy[0],
                                "minY": xyxy[1],
                                "maxX": xyxy[2],
                                "maxY": xyxy[3],
                            },
                            "class_id": int(cls),
                            "box_caption": "%s %.3f" % (names[cls], conf),
                            "scores": {"class_score": conf},
                            "domain": "pixel",
                        }
                        for *xyxy, conf, cls in pred.tolist()
                    ]
                    boxes = {
                        "predictions": {"box_data": box_data, "class_labels": names}
                    }  # inference-space
                    wandb_images.append(
                        wandb_logger.wandb.Image(
                            img[out_idx], boxes=boxes, caption=path.name
                        )
                    )
            wandb_logger.log_training_progress(
                predn, path, names
            ) if wandb_logger and wandb_logger.wandb_run else None
            """

            # Append to pycocotools JSON dictionary
            """if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append(
                        {
                            "image_id": image_id,
                            "category_id": coco91class[int(p[5])]
                            if is_coco
                            else int(p[5]),
                            "bbox": [round(x, 3) for x in b],
                            "score": round(p[4], 5),
                        }
                    )"""

            """# Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if num_labels:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(
                    img[out_idx].shape[1:], tbox, shapes[out_idx][0], shapes[out_idx][1]
                )  # native-space labels
                if plots:
                    confusion_matrix.process_batch(
                        predn, torch.cat((labels[:, 0:1], tbox), 1)
                    )

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (
                        (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    )  # prediction indices
                    pi = (
                        (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                    )  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(
                            1
                        )  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if (
                                    len(detected) == num_labels
                                ):  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))"""

    def _plot_results(
        self,
        batch_idx: int,
        img: Tensor,
        out: Tensor,
        targets: Tensor,
        paths: List[str],
    ):
        f = self._save_dir / f"test_batch{batch_idx}_labels.jpg"  # labels
        Thread(
            target=plot_images,
            args=(img, targets, paths, f, self._model.names),
            daemon=True,
        ).start()
        f = self._save_dir / f"test_batch{batch_idx}_pred.jpg"  # predictions
        Thread(
            target=plot_images,
            args=(img, output_to_target(out), paths, f, self._model.names),
            daemon=True,
        ).start()

    def test(self):
        tqdm_head = ("%20s" + "%12s" * 6) % (
            "Class",
            "Images",
            "Labels",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        for batch_idx, (img, targets, paths, shapes) in enumerate(
            tqdm(self._test_dataloader, desc=tqdm_head)
        ):
            self._single_pass(batch_idx, img, targets, paths, shapes)

    def speed(self):
        self._plot = False
        self._log_stats = False

        self._conf_threshold = 0.25
        self._iou_threshold = 0.45

        self.test()

    def study(self, parameter_space=None):
        raise NotImplementedError
        pass
