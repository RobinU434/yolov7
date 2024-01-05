from argparse import ArgumentParser


def add_export_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--model-path",
        help="weights path",
        dest="model_path",
        type=str,
    )
    parser.add_argument(
        "--image-size",
        help="image size. Defaults to [640, 400].",
        dest="image_size",
        type=int,
        nargs="+",
        default=[640, 640],
    )
    parser.add_argument(
        "--max-wh",
        help="None for tensorrt nms, int value for onnx-runtime nms. Defaults to 64.",
        dest="max_wh",
        type=int,
        default="64",
    )
    parser.add_argument(
        "--grid",
        help="export Detect() layer grid. Defaults to False.",
        dest="grid",
        type=bool,
        default="False",
    )
    parser.add_argument(
        "--end2end",
        help="export end2end onnx. Defaults to False.",
        dest="end2end",
        type=bool,
        default="False",
    )
    parser.add_argument(
        "--simplify",
        help="simplify onnx model. Defaults to False.",
        dest="simplify",
        type=bool,
        default="False",
    )
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
