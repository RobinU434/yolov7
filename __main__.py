from argparse import ArgumentParser
from yolov7.process.process import YOLOV7Process
from yolov7.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = YOLOV7Process()
    match args["command"]:
        case "download-data":
            module.download_data(
                source=args["source"],
                destination=args["destination"],
                unpack=args["unpack"],
            )

        case "train":
            module.train()

        case "test":
            module.test()

        case "export":
            module.export()

        case _:
            return False

    return True


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="process to handle a YOLOV7 model")

    parser = setup_parser(parser)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    if not execute(args_dict):
        parser.print_usage()


if __name__ == "__main__":
    main()
