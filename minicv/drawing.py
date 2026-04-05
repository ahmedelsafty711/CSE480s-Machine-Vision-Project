"""
minicv.drawing
==============
Drawing primitives that operate directly on NumPy arrays.

All functions mutate the image IN-PLACE and return it for chaining.

Functions
---------
draw_point(image, x, y, color, radius)
draw_line(image, x0, y0, x1, y1, color, thickness)
draw_rectangle(image, x0, y0, x1, y1, color, thickness, filled)
draw_polygon(image, vertices, color, thickness, filled)
put_text(image, text, x, y, font_scale, color)

Color format
------------
- Grayscale image (H, W)     : scalar int/float (e.g. 255 or 0.5).
- RGB image       (H, W, 3)  : 3-tuple (R, G, B) with values matching image range.

All coordinates are in (x, y) image convention (x = column, y = row).
All functions clip drawing to the canvas boundary silently.
"""

from __future__ import annotations

import math
import numpy as np

from .utils import _check_image_2d_or_3d


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _apply_color(image: np.ndarray, y: int, x: int, color) -> None:
    """Write *color* to pixel (y, x) clipping to canvas."""
    H, W = image.shape[:2]
    if 0 <= y < H and 0 <= x < W:
        if image.ndim == 2:
            image[y, x] = color
        else:
            image[y, x] = color


def _validate_color(image: np.ndarray, color) -> None:
    if image.ndim == 2:
        if not isinstance(color, (int, float, np.integer, np.floating)):
            raise TypeError(
                "For grayscale images, color must be a scalar (int or float)."
            )
    else:
        if not (hasattr(color, "__len__") and len(color) == 3):
            raise TypeError(
                "For RGB images, color must be a 3-element tuple/list (R, G, B)."
            )


# ---------------------------------------------------------------------------
# Section 7 – Drawing Primitives
# ---------------------------------------------------------------------------

def draw_point(
    image: np.ndarray,
    x: int,
    y: int,
    color=255,
    radius: int = 1,
) -> np.ndarray:
    """
    Draw a filled circle (point) on an image.

    Parameters
    ----------
    image : np.ndarray
        Target image, modified in-place. Shape (H, W) or (H, W, 3).
    x : int
        Horizontal coordinate of the point centre (column index).
    y : int
        Vertical coordinate of the point centre (row index).
    color : scalar or 3-tuple
        Drawing colour. Scalar for grayscale; (R, G, B) tuple for RGB.
    radius : int, optional
        Radius of the point in pixels. Default 1.

    Returns
    -------
    np.ndarray
        The modified image (same object as *image*).

    Raises
    ------
    TypeError  : Incompatible color format.
    ValueError : *radius* < 1.
    """
    _check_image_2d_or_3d(image)
    _validate_color(image, color)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}.")

    H, W = image.shape[:2]
    for dy in range(-radius + 1, radius):
        for dx in range(-radius + 1, radius):
            if dx * dx + dy * dy < radius * radius:
                _apply_color(image, y + dy, x + dx, color)
    return image


def draw_line(
    image: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color=255,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a line between two points using Bresenham's algorithm.

    Parameters
    ----------
    image : np.ndarray
        Target image, modified in-place.
    x0, y0 : int
        Start point (column, row).
    x1, y1 : int
        End point (column, row).
    color : scalar or 3-tuple
        Drawing colour.
    thickness : int, optional
        Line width in pixels. Default 1.

    Returns
    -------
    np.ndarray
        The modified image.

    Raises
    ------
    TypeError  : Incompatible color format.
    ValueError : *thickness* < 1.

    Notes
    -----
    Bresenham's line algorithm iterates over the dominant axis and sets
    the appropriate pixel at each step — O(max(|Δx|, |Δy|)) pixel writes
    with integer arithmetic only.  Thickness is approximated by drawing
    parallel offset lines.
    """
    _check_image_2d_or_3d(image)
    _validate_color(image, color)
    if thickness < 1:
        raise ValueError(f"thickness must be >= 1, got {thickness}.")

    def _bresenham(x0, y0, x1, y1):
        """Yield all (x, y) pixels along the Bresenham line."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    offsets = range(-(thickness // 2), thickness // 2 + 1)
    for px, py in _bresenham(x0, y0, x1, y1):
        for o in offsets:
            if abs(x1 - x0) >= abs(y1 - y0):
                _apply_color(image, py + o, px, color)
            else:
                _apply_color(image, py, px + o, color)
    return image


def draw_rectangle(
    image: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color=255,
    thickness: int = 1,
    filled: bool = False,
) -> np.ndarray:
    """
    Draw a rectangle on the image.

    Parameters
    ----------
    image : np.ndarray
        Target image, modified in-place.
    x0, y0 : int
        Top-left corner (column, row).
    x1, y1 : int
        Bottom-right corner (column, row).
    color : scalar or 3-tuple
        Drawing colour.
    thickness : int, optional
        Border thickness for outline mode. Default 1.
    filled : bool, optional
        If True, the rectangle interior is filled. Default False.

    Returns
    -------
    np.ndarray
        The modified image.
    """
    _check_image_2d_or_3d(image)
    _validate_color(image, color)

    H, W = image.shape[:2]
    rx0, rx1 = max(0, min(x0, x1)), min(W - 1, max(x0, x1))
    ry0, ry1 = max(0, min(y0, y1)), min(H - 1, max(y0, y1))

    if filled:
        if image.ndim == 2:
            image[ry0:ry1 + 1, rx0:rx1 + 1] = color
        else:
            image[ry0:ry1 + 1, rx0:rx1 + 1] = color
    else:
        draw_line(image, rx0, ry0, rx1, ry0, color, thickness)  # top
        draw_line(image, rx0, ry1, rx1, ry1, color, thickness)  # bottom
        draw_line(image, rx0, ry0, rx0, ry1, color, thickness)  # left
        draw_line(image, rx1, ry0, rx1, ry1, color, thickness)  # right

    return image


def draw_polygon(
    image: np.ndarray,
    vertices: list[tuple[int, int]],
    color=255,
    thickness: int = 1,
    filled: bool = False,
) -> np.ndarray:
    """
    Draw a polygon defined by a list of vertices.

    Parameters
    ----------
    image : np.ndarray
        Target image, modified in-place.
    vertices : list of (x, y) int tuples
        Polygon vertices in order. The polygon is automatically closed.
    color : scalar or 3-tuple
        Drawing colour.
    thickness : int, optional
        Edge thickness for outline mode. Default 1.
    filled : bool, optional
        If True, fills the polygon interior using a scanline algorithm.
        Default False.

    Returns
    -------
    np.ndarray
        The modified image.

    Raises
    ------
    ValueError : If *vertices* has fewer than 3 points.

    Notes
    -----
    Filled polygon uses a scanline rasterisation: for each row, find
    edge intersections, sort them, and fill between pairs.
    """
    _check_image_2d_or_3d(image)
    _validate_color(image, color)
    if len(vertices) < 3:
        raise ValueError(
            f"A polygon requires at least 3 vertices, got {len(vertices)}."
        )

    n = len(vertices)
    edges = [(vertices[i], vertices[(i + 1) % n]) for i in range(n)]

    if filled:
        H, W = image.shape[:2]
        ys = [v[1] for v in vertices]
        y_min, y_max = max(0, min(ys)), min(H - 1, max(ys))

        for y in range(y_min, y_max + 1):
            xs_cross = []
            for (ax, ay), (bx, by) in edges:
                if ay == by:
                    continue
                if min(ay, by) <= y < max(ay, by):
                    x_int = ax + (y - ay) * (bx - ax) / (by - ay)
                    xs_cross.append(x_int)
            xs_cross.sort()
            for i in range(0, len(xs_cross) - 1, 2):
                xi0 = max(0, int(math.floor(xs_cross[i])))
                xi1 = min(W - 1, int(math.ceil(xs_cross[i + 1])))
                if image.ndim == 2:
                    image[y, xi0:xi1 + 1] = color
                else:
                    image[y, xi0:xi1 + 1] = color
    else:
        for (ax, ay), (bx, by) in edges:
            draw_line(image, ax, ay, bx, by, color, thickness)

    return image


# ---------------------------------------------------------------------------
# Section 8 – Text Placement
# ---------------------------------------------------------------------------

# Minimal 5×7 pixel font (ASCII 32–126).  Each character is a list of 7 rows;
# each row is a 5-bit integer (MSB = leftmost pixel).
_FONT_W = 5
_FONT_H = 7

_FONT: dict[str, list[int]] = {
    " ": [0b00000] * 7,
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "F": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "I": [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "J": [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "M": [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "Q": [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "S": [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    "W": [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
    "X": [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001],
    "Y": [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "Z": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
    "3": [0b11111, 0b00001, 0b00010, 0b00110, 0b00001, 0b10001, 0b01110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    ".": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
    ",": [0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100, 0b01000],
    "!": [0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00000, 0b00100],
    "?": [0b01110, 0b10001, 0b00001, 0b00110, 0b00100, 0b00000, 0b00100],
    ":": [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
    "-": [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
    "_": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
    "/": [0b00001, 0b00010, 0b00100, 0b00100, 0b01000, 0b10000, 0b10000],
    "(": [0b00100, 0b01000, 0b10000, 0b10000, 0b10000, 0b01000, 0b00100],
    ")": [0b00100, 0b00010, 0b00001, 0b00001, 0b00001, 0b00010, 0b00100],
}


def put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float = 1.0,
    color=255,
) -> np.ndarray:
    """
    Render ASCII text onto an image at position (*x*, *y*).

    Parameters
    ----------
    image : np.ndarray
        Target image, modified in-place.
    text : str
        The string to render. Unsupported characters are replaced by spaces.
    x : int
        Left edge column of the first character.
    y : int
        Top edge row of the first character.
    font_scale : float, optional
        Scale factor applied to the built-in 5×7 pixel font.
        1.0 = 5×7; 2.0 = 10×14; etc. Default 1.0.
    color : scalar or 3-tuple
        Drawing colour.

    Returns
    -------
    np.ndarray
        The modified image.

    Raises
    ------
    TypeError  : If *text* is not a str or types are wrong.
    ValueError : If *font_scale* <= 0.
    """
    _check_image_2d_or_3d(image)
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}.")
    _validate_color(image, color)
    if font_scale <= 0:
        raise ValueError(f"font_scale must be > 0, got {font_scale}.")

    scale = max(1, int(round(font_scale)))
    char_w = _FONT_W * scale
    char_h = _FONT_H * scale

    H, W = image.shape[:2]
    cx = x

    for ch in text.upper():
        bitmap = _FONT.get(ch, _FONT[" "])
        for row_i, row_bits in enumerate(bitmap):
            for col_i in range(_FONT_W):
                bit = (row_bits >> (_FONT_W - 1 - col_i)) & 1
                if bit:
                    py_start = y + row_i * scale
                    px_start = cx + col_i * scale
                    for dy in range(scale):
                        for dx in range(scale):
                            _apply_color(image, py_start + dy, px_start + dx, color)
        cx += char_w + scale   # 1-pixel gap between chars

    return image
