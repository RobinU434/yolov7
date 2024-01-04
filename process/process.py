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
        pass

    def download_data(self, source: str, destination: str, unpack: bool = True):
        """
        download data

        Args:
            source (List[str]): file with specified urls where to download the zips from
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
        """
        start training process
        """
        pass

    def test(self, data_path: str):
        """
        start test process
        """
        pass

    def export(self):
        """
        export model
        """
        pass
