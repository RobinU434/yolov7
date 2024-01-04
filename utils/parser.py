from argparse import ArgumentParser


def add_export_args(parser: ArgumentParser) -> ArgumentParser:
    return parser


def add_test_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--data-path",
        help="_description_",
        dest="data_path",
        type=str,
    )
    parser.add_argument(
        "--model-path",
        help="path to model checkpoint",
        dest="model_path",
        type=str,
    )
    parser.add_argument(
        "--task",
        help="test, speed or study",
        dest="task",
        type=str,
        default="test",
    )
    return parser


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--data-path",
        help="_description_",
        dest="data_path",
        type=str,
    )
    return parser


def add_download_data_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--source",
        help="file with specified urls where to download the zips from",
        dest="source",
        type=str,
    )
    parser.add_argument(
        "--destination",
        help="where to store the downloaded files",
        dest="destination",
        type=str,
    )
    parser.add_argument(
        "--unpack",
        help="Would you like to unpack the zip files. Default set to True",
        dest="unpack",
        type=bool,
        default="True",
    )
    return parser


def setup_yolov7process_parser(parser: ArgumentParser) -> ArgumentParser:
    command_subparser = parser.add_subparsers(dest="command", title="command")
    download_data = command_subparser.add_parser("download-data", help="download data")
    download_data = add_download_data_args(download_data)
    train = command_subparser.add_parser("train", help="start training process")
    train = add_train_args(train)
    test = command_subparser.add_parser("test", help="start test process")
    test = add_test_args(test)
    export = command_subparser.add_parser("export", help="export model")
    export = add_export_args(export)
    return parser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser = setup_yolov7process_parser(parser)
    return parser
