import datetime
from pathlib import Path


def date_string(path: str = __file__) -> str:
    """return human-readable file modification date
    i.e. '2021-3-26'

    Args:
        path (str, optional): path to file. Defaults to __file__.

    Returns:
        str: date string of modification date
    """
    # return human-readable file modification date,
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"
