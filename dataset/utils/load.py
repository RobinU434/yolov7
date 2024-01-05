from torch.utils.data import DataLoader
from yolov7.dataset.datasets import create_dataloader
from yolov7.dataset.utils.checks import check_img_size
from yolov7.utils.file_system import load_yaml
from yolov7.utils.print import colorstr


def load_coco_dataloader(
    data_path: str = "yolov7",
    entity: str = "train",
    batch_size: int = 32,
    image_size: int = 640,
    grid_stride: int = 32,
    single_cls: bool = False,
) -> DataLoader:
    entity = entity.lower()
    assert entity in ["train", "val", "test"]

    coco_meta = load_yaml(data_path + "data/coco.yaml")
    entity_images = data_path + coco_meta[entity]

    image_size = check_img_size(image_size, grid_stride)
    dataloader, _ = create_dataloader(
        path=entity_images,
        imgsz=image_size,
        batch_size=batch_size,
        stride=grid_stride,
        single_cls=single_cls,
        pad=0.5,
        rect=True,
        prefix=colorstr(f"{entity}: "),
    )

    return dataloader
