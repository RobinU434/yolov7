from yolov7.dataset.utils import load_dataset
from yolov7.models.utils import load_model
from yolov7.utils.file_system import increment_path
from yolov7.utils.print import print_status
from yolov7.utils.torch import check_batch_size, select_device


class Tester:
    def __init__(
        self,
        model_path: str,
        data_path: str,
        batch_size: int = 32,
        device: str = "cpu",
        save_dir: str = "runs/test/exp",
        half_precision: bool = True,
    ) -> None:
        self._device = select_device(device)
        self._model = load_model(
            model_path,
        )

        if self._device.type != "cpu" and half_precision:
            self._model.half()
        self._model.eval()

        self._batch_size = check_batch_size(self._device, batch_size)
        self._test_data = load_dataset(data_path, entity="test")

        self._save_dir = increment_path(save_dir, exist_ok=False)
        self._save_dir.mkdir(parents=True, exist_ok=False)

        print_status(self._device)

    def test(self):
        pass

    def speed(self):
        pass

    def study(self):
        pass
