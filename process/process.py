import logging
from typing import List
from yolov7.dataset.utils.checks import check_dataset_coco
from yolov7.process.export import Exporter
from yolov7.process.tester import Tester
from yolov7.utils.file_system import (
    check_existing_directory,
    create_directory,
    download_zip,
    load_txt,
    unpack_zip,
)


class YOLOV7Process:
    """
    process to handle a YOLOV7 model
    """

    def __init__(self) -> None:
        """init class"""

    def download_data(self, source: str, destination: str, unpack: bool = True):
        """
        download data

        Args:
            source (str): file with specified urls where to download the zips from
            destination (str): where to store the downloaded files
            unpack (bool): Would you like to unpack the zip files. Default set to True
        """
        urls = load_txt(source)

        if not check_existing_directory(destination):
            create_directory(destination)

        zip_paths = download_zip(urls, destination)

        if unpack:
            unpack_zip(zip_paths)

    def train(self, data_path: str):
        """start training process

        Args:
            data_path (str): _description_
        """
        if not check_dataset_coco(data_path):
            msg = "Not possible to continue without proper data source. Please download the data with the 'download' command provided by this toolbox."
            logging.fatal(msg)
            exit()

    def test(self, data_path: str, model_path: str, task: str = "test"):
        """
        start test process

        Args:
            data_path (str): _description_
            model_path (str): path to model checkpoint
            task (str): test, speed or study
        """
        tester = Tester(model_path, data_path)

        match task:
            case "test":
                tester.test()
            case "speed":
                tester.speed()
            case "study":
                tester.study()

    def export(
        self,
        model_path: str,
        image_size: list = [640, 640],
        max_wh: int = 64,
        grid: bool = False,
        end2end: bool = False,
        simplify: bool = False,
    ):
        """export model

        Args:
            model_path (str): weights path
            image_size (List[int], optional): image size. Defaults to [640, 400].
            max_wh (int): None for tensorrt nms, int value for onnx-runtime nms. Defaults to 64.
            grid (bool, optional): export Detect() layer grid. Defaults to False.
            end2end (bool, optional): export end2end onnx. Defaults to False.
            simplify (bool, optional): simplify onnx model. Defaults to False.
        """
        print(image_size)
        exporter = Exporter(
            image_size=image_size,
            grid=grid,
            end2end=end2end,
            simplify=simplify,
            max_wh=max_wh,
        )

        exporter.export(model_path=model_path)
