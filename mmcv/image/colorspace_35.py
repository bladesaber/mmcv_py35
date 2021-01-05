
import cv2
import numpy as np


def imconvert(img, src, dst):
    "Convert an image from the src colorspace to dst colorspace.\n\n    Args:\n        img (ndarray): The input image.\n        src (str): The source colorspace, e.g., 'rgb', 'hsv'.\n        dst (str): The destination colorspace, e.g., 'rgb', 'hsv'.\n\n    Returns:\n        ndarray: The converted image.\n    "
    code = getattr(cv2, ''.join(['COLOR_', '{}'.format(
        src.upper()), '2', '{}'.format(dst.upper())]))
    out_img = cv2.cvtColor(img, code)
    return out_img


def bgr2gray(img, keepdim=False):
    'Convert a BGR image to grayscale image.\n\n    Args:\n        img (ndarray): The input image.\n        keepdim (bool): If False (by default), then return the grayscale image\n            with 2 dims, otherwise 3 dims.\n\n    Returns:\n        ndarray: The converted grayscale image.\n    '
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[(..., None)]
    return out_img


def rgb2gray(img, keepdim=False):
    'Convert a RGB image to grayscale image.\n\n    Args:\n        img (ndarray): The input image.\n        keepdim (bool): If False (by default), then return the grayscale image\n            with 2 dims, otherwise 3 dims.\n\n    Returns:\n        ndarray: The converted grayscale image.\n    '
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if keepdim:
        out_img = out_img[(..., None)]
    return out_img


def gray2bgr(img):
    'Convert a grayscale image to BGR image.\n\n    Args:\n        img (ndarray): The input image.\n\n    Returns:\n        ndarray: The converted BGR image.\n    '
    img = (img[(..., None)] if (img.ndim == 2) else img)
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img


def gray2rgb(img):
    'Convert a grayscale image to RGB image.\n\n    Args:\n        img (ndarray): The input image.\n\n    Returns:\n        ndarray: The converted RGB image.\n    '
    img = (img[(..., None)] if (img.ndim == 2) else img)
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return out_img


def _convert_input_type_range(img):
    'Convert the type and range of the input image.\n\n    It converts the input image to np.float32 type and range of [0, 1].\n    It is mainly used for pre-processing the input image in colorspace\n    convertion functions such as rgb2ycbcr and ycbcr2rgb.\n\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n\n    Returns:\n        (ndarray): The converted image with type of np.float32 and range of\n            [0, 1].\n    '
    img_type = img.dtype
    img = img.astype(np.float32)
    if (img_type == np.float32):
        pass
    elif (img_type == np.uint8):
        img /= 255.0
    else:
        raise TypeError(''.join(
            ['The img type should be np.float32 or np.uint8, but got ', '{}'.format(img_type)]))
    return img


def _convert_output_type_range(img, dst_type):
    'Convert the type and range of the image according to dst_type.\n\n    It converts the image to desired type and range. If `dst_type` is np.uint8,\n    images will be converted to np.uint8 type with range [0, 255]. If\n    `dst_type` is np.float32, it converts the image to np.float32 type with\n    range [0, 1].\n    It is mainly used for post-processing images in colorspace convertion\n    functions such as rgb2ycbcr and ycbcr2rgb.\n\n    Args:\n        img (ndarray): The image to be converted with np.float32 type and\n            range [0, 255].\n        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it\n            converts the image to np.uint8 type with range [0, 255]. If\n            dst_type is np.float32, it converts the image to np.float32 type\n            with range [0, 1].\n\n    Returns:\n        (ndarray): The converted image with desired type and range.\n    '
    if (dst_type not in (np.uint8, np.float32)):
        raise TypeError(''.join(
            ['The dst_type should be np.float32 or np.uint8, but got ', '{}'.format(dst_type)]))
    if (dst_type == np.uint8):
        img = img.round()
    else:
        img /= 255.0
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    "Convert a RGB image to YCbCr image.\n\n    This function produces the same results as Matlab's `rgb2ycbcr` function.\n    It implements the ITU-R BT.601 conversion for standard-definition\n    television. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.\n\n    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.\n    In OpenCV, it implements a JPEG conversion. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.\n\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n        y_only (bool): Whether to only return Y channel. Default: False.\n\n    Returns:\n        ndarray: The converted YCbCr image. The output image has the same type\n            and range as input image.\n    "
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = (np.dot(img, [65.481, 128.553, 24.966]) + 16.0)
    else:
        out_img = (np.matmul(img, [[65.481, (- 37.797), 112.0], [
                   128.553, (- 74.203), (- 93.786)], [24.966, 112.0, (- 18.214)]]) + [16, 128, 128])
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    'Convert a BGR image to YCbCr image.\n\n    The bgr version of rgb2ycbcr.\n    It implements the ITU-R BT.601 conversion for standard-definition\n    television. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.\n\n    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.\n    In OpenCV, it implements a JPEG conversion. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.\n\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n        y_only (bool): Whether to only return Y channel. Default: False.\n\n    Returns:\n        ndarray: The converted YCbCr image. The output image has the same type\n            and range as input image.\n    '
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = (np.dot(img, [24.966, 128.553, 65.481]) + 16.0)
    else:
        out_img = (np.matmul(img, [[24.966, 112.0, (- 18.214)], [
                   128.553, (- 74.203), (- 93.786)], [65.481, (- 37.797), 112.0]]) + [16, 128, 128])
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    "Convert a YCbCr image to RGB image.\n\n    This function produces the same results as Matlab's ycbcr2rgb function.\n    It implements the ITU-R BT.601 conversion for standard-definition\n    television. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.\n\n    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.\n    In OpenCV, it implements a JPEG conversion. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.\n\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n\n    Returns:\n        ndarray: The converted RGB image. The output image has the same type\n            and range as input image.\n    "
    img_type = img.dtype
    img = (_convert_input_type_range(img) * 255)
    out_img = ((np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [
               0, (- 0.00153632), 0.00791071], [0.00625893, (- 0.00318811), 0]]) * 255.0) + [(- 222.921), 135.576, (- 276.836)])
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    'Convert a YCbCr image to BGR image.\n\n    The bgr version of ycbcr2rgb.\n    It implements the ITU-R BT.601 conversion for standard-definition\n    television. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.\n\n    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.\n    In OpenCV, it implements a JPEG conversion. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.\n\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n\n    Returns:\n        ndarray: The converted BGR image. The output image has the same type\n            and range as input image.\n    '
    img_type = img.dtype
    img = (_convert_input_type_range(img) * 255)
    out_img = ((np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [
               0.00791071, (- 0.00153632), 0], [0, (- 0.00318811), 0.00625893]]) * 255.0) + [(- 276.836), 135.576, (- 222.921)])
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def convert_color_factory(src, dst):
    code = getattr(cv2, ''.join(['COLOR_', '{}'.format(
        src.upper()), '2', '{}'.format(dst.upper())]))

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img
    convert_color.__doc__ = ''.join(['Convert a ', '{}'.format(src.upper()), ' image to ', '{}'.format(dst.upper(
    )), '\n        image.\n\n    Args:\n        img (ndarray or str): The input image.\n\n    Returns:\n        ndarray: The converted ', '{}'.format(dst.upper()), ' image.\n    '])
    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')
rgb2bgr = convert_color_factory('rgb', 'bgr')
bgr2hsv = convert_color_factory('bgr', 'hsv')
hsv2bgr = convert_color_factory('hsv', 'bgr')
bgr2hls = convert_color_factory('bgr', 'hls')
hls2bgr = convert_color_factory('hls', 'bgr')
