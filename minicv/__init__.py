"""
minicv – A minimal OpenCV-like image-processing library.
Implemented from scratch using only NumPy, Pandas, Matplotlib,
and the Python standard library.

Modules
-------
io          : read_image, export_image
utils       : rgb_to_gray, gray_to_rgb, normalize, clip_pixels, pad_image
filtering   : convolve2d, spatial_filter, mean_filter, gaussian_kernel,
              gaussian_filter, median_filter, threshold_global,
              threshold_otsu, threshold_adaptive, sobel_gradients
processing  : bit_plane_slice, histogram, histogram_equalization,
              unsharp_mask, morphological_op
transforms  : resize, rotate, translate
features    : color_histogram, hu_moments, hog, lbp, gradient_hist
drawing     : draw_point, draw_line, draw_rectangle, draw_polygon, put_text
"""

from .io import read_image, export_image
from .utils import (
    rgb_to_gray, gray_to_rgb,
    normalize, clip_pixels, pad_image,
)
from .filtering import (
    convolve2d, spatial_filter,
    mean_filter, gaussian_kernel, gaussian_filter,
    median_filter,
    threshold_global, threshold_otsu, threshold_adaptive,
    sobel_gradients,
)
from .processing import (
    bit_plane_slice, histogram, histogram_equalization,
    unsharp_mask, morphological_op,
)
from .transforms import resize, rotate, translate
from .features import color_histogram, hu_moments, hog, lbp, gradient_hist
from .drawing import (
    draw_point, draw_line, draw_rectangle, draw_polygon, put_text,
)

__version__ = "1.0.0"
__all__ = [
    # io
    "read_image", "export_image",
    # utils
    "rgb_to_gray", "gray_to_rgb", "normalize", "clip_pixels", "pad_image",
    # filtering
    "convolve2d", "spatial_filter",
    "mean_filter", "gaussian_kernel", "gaussian_filter",
    "median_filter",
    "threshold_global", "threshold_otsu", "threshold_adaptive",
    "sobel_gradients",
    # processing
    "bit_plane_slice", "histogram", "histogram_equalization",
    "unsharp_mask", "morphological_op",
    # transforms
    "resize", "rotate", "translate",
    # features
    "color_histogram", "hu_moments", "hog", "lbp", "gradient_hist",
    # drawing
    "draw_point", "draw_line", "draw_rectangle", "draw_polygon", "put_text",
]
