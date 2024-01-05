from typing import List, Tuple
import matplotlib


def rgb_color_list() -> List[Tuple[int, int, int]]:
    """Return first 10 matplotlib.pyplot colors as (r,g,b)
    https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb

    Returns:
        List[Tuple[int, int, int]]: rgb colors
    """

    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    return [
        hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()
    ]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)
