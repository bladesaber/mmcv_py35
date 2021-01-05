
import numbers
import cv2
import numpy as np
from .io import imread_backend
try:
    from PIL import Image
except ImportError:
    Image = None


def _scale_size(size, scale):
    'Rescale a size by a ratio.\n\n    Args:\n        size (tuple[int]): (w, h).\n        scale (float): Scaling factor.\n\n    Returns:\n        tuple[int]: scaled size.\n    '
    (w, h) = size
    return (int(((w * float(scale)) + 0.5)), int(((h * float(scale)) + 0.5)))


cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}
if (Image is not None):
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING,
    }


def imresize(img, size, return_scale=False, interpolation='bilinear', out=None, backend=None):
    'Resize image to a given size.\n\n    Args:\n        img (ndarray): The input image.\n        size (tuple[int]): Target size (w, h).\n        return_scale (bool): Whether to return `w_scale` and `h_scale`.\n        interpolation (str): Interpolation method, accepted values are\n            "nearest", "bilinear", "bicubic", "area", "lanczos" for \'cv2\'\n            backend, "nearest", "bilinear" for \'pillow\' backend.\n        out (ndarray): The output destination.\n        backend (str | None): The image resize backend type. Options are `cv2`,\n            `pillow`, `None`. If backend is None, the global imread_backend\n            specified by ``mmcv.use_backend()`` will be used. Default: None.\n\n    Returns:\n        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or\n            `resized_img`.\n    '
    (h, w) = img.shape[:2]
    if (backend is None):
        backend = imread_backend
    if (backend not in ['cv2', 'pillow']):
        raise ValueError(''.join(['backend: ', '{}'.format(
            backend), " is not supported for resize.Supported backends are 'cv2', 'pillow'"]))
    if (backend == 'pillow'):
        assert (img.dtype == np.uint8), 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if (not return_scale):
        return resized_img
    else:
        w_scale = (size[0] / w)
        h_scale = (size[1] / h)
        return (resized_img, w_scale, h_scale)


def imresize_like(img, dst_img, return_scale=False, interpolation='bilinear', backend=None):
    'Resize image to the same size of a given image.\n\n    Args:\n        img (ndarray): The input image.\n        dst_img (ndarray): The target image.\n        return_scale (bool): Whether to return `w_scale` and `h_scale`.\n        interpolation (str): Same as :func:`resize`.\n        backend (str | None): Same as :func:`resize`.\n\n    Returns:\n        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or\n            `resized_img`.\n    '
    (h, w) = dst_img.shape[:2]
    return imresize(img, (w, h), return_scale, interpolation, backend=backend)


def rescale_size(old_size, scale, return_scale=False):
    'Calculate the new size to be rescaled to.\n\n    Args:\n        old_size (tuple[int]): The old size (w, h) of image.\n        scale (float | tuple[int]): The scaling factor or maximum size.\n            If it is a float number, then the image will be rescaled by this\n            factor, else if it is a tuple of 2 integers, then the image will\n            be rescaled as large as possible within the scale.\n        return_scale (bool): Whether to return the scaling factor besides the\n            rescaled image size.\n\n    Returns:\n        tuple[int]: The new rescaled image size.\n    '
    (w, h) = old_size
    if isinstance(scale, (float, int)):
        if (scale <= 0):
            raise ValueError(
                ''.join(['Invalid scale ', '{}'.format(scale), ', must be positive.']))
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min((max_long_edge / max(h, w)),
                           (max_short_edge / min(h, w)))
    else:
        raise TypeError(''.join(
            ['Scale must be a number or tuple of int, but got ', '{}'.format(type(scale))]))
    new_size = _scale_size((w, h), scale_factor)
    if return_scale:
        return (new_size, scale_factor)
    else:
        return new_size


def imrescale(img, scale, return_scale=False, interpolation='bilinear', backend=None):
    'Resize image while keeping the aspect ratio.\n\n    Args:\n        img (ndarray): The input image.\n        scale (float | tuple[int]): The scaling factor or maximum size.\n            If it is a float number, then the image will be rescaled by this\n            factor, else if it is a tuple of 2 integers, then the image will\n            be rescaled as large as possible within the scale.\n        return_scale (bool): Whether to return the scaling factor besides the\n            rescaled image.\n        interpolation (str): Same as :func:`resize`.\n        backend (str | None): Same as :func:`resize`.\n\n    Returns:\n        ndarray: The rescaled image.\n    '
    (h, w) = img.shape[:2]
    (new_size, scale_factor) = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return (rescaled_img, scale_factor)
    else:
        return rescaled_img


def imflip(img, direction='horizontal'):
    'Flip an image horizontally or vertically.\n\n    Args:\n        img (ndarray): Image to be flipped.\n        direction (str): The flip direction, either "horizontal" or\n            "vertical" or "diagonal".\n\n    Returns:\n        ndarray: The flipped image.\n    '
    assert (direction in ['horizontal', 'vertical', 'diagonal'])
    if (direction == 'horizontal'):
        return np.flip(img, axis=1)
    elif (direction == 'vertical'):
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def imflip_(img, direction='horizontal'):
    'Inplace flip an image horizontally or vertically.\n\n    Args:\n        img (ndarray): Image to be flipped.\n        direction (str): The flip direction, either "horizontal" or\n            "vertical" or "diagonal".\n\n    Returns:\n        ndarray: The flipped image (inplace).\n    '
    assert (direction in ['horizontal', 'vertical', 'diagonal'])
    if (direction == 'horizontal'):
        return cv2.flip(img, 1, img)
    elif (direction == 'vertical'):
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, (- 1), img)


def imrotate(img, angle, center=None, scale=1.0, border_value=0, interpolation='bilinear', auto_bound=False):
    'Rotate an image.\n\n    Args:\n        img (ndarray): Image to be rotated.\n        angle (float): Rotation angle in degrees, positive values mean\n            clockwise rotation.\n        center (tuple[float], optional): Center point (w, h) of the rotation in\n            the source image. If not specified, the center of the image will be\n            used.\n        scale (float): Isotropic scale factor.\n        border_value (int): Border value.\n        interpolation (str): Same as :func:`resize`.\n        auto_bound (bool): Whether to adjust the image size to cover the whole\n            rotated image.\n\n    Returns:\n        ndarray: The rotated image.\n    '
    if ((center is not None) and auto_bound):
        raise ValueError('`auto_bound` conflicts with `center`')
    (h, w) = img.shape[:2]
    if (center is None):
        center = (((w - 1) * 0.5), ((h - 1) * 0.5))
    assert isinstance(center, tuple)
    matrix = cv2.getRotationMatrix2D(center, (- angle), scale)
    if auto_bound:
        cos = np.abs(matrix[(0, 0)])
        sin = np.abs(matrix[(0, 1)])
        new_w = ((h * sin) + (w * cos))
        new_h = ((h * cos) + (w * sin))
        matrix[(0, 2)] += ((new_w - w) * 0.5)
        matrix[(1, 2)] += ((new_h - h) * 0.5)
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img, matrix, (w, h), flags=cv2_interp_codes[interpolation], borderValue=border_value)
    return rotated


def bbox_clip(bboxes, img_shape):
    'Clip bboxes to fit the image shape.\n\n    Args:\n        bboxes (ndarray): Shape (..., 4*k)\n        img_shape (tuple[int]): (height, width) of the image.\n\n    Returns:\n        ndarray: Clipped bboxes.\n    '
    assert ((bboxes.shape[(- 1)] % 4) == 0)
    cmin = np.empty(bboxes.shape[(- 1)], dtype=bboxes.dtype)
    cmin[0::2] = (img_shape[1] - 1)
    cmin[1::2] = (img_shape[0] - 1)
    clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
    return clipped_bboxes


def bbox_scaling(bboxes, scale, clip_shape=None):
    'Scaling bboxes w.r.t the box center.\n\n    Args:\n        bboxes (ndarray): Shape(..., 4).\n        scale (float): Scaling factor.\n        clip_shape (tuple[int], optional): If specified, bboxes that exceed the\n            boundary will be clipped according to the given shape (h, w).\n\n    Returns:\n        ndarray: Scaled bboxes.\n    '
    if (float(scale) == 1.0):
        scaled_bboxes = bboxes.copy()
    else:
        w = ((bboxes[(..., 2)] - bboxes[(..., 0)]) + 1)
        h = ((bboxes[(..., 3)] - bboxes[(..., 1)]) + 1)
        dw = ((w * (scale - 1)) * 0.5)
        dh = ((h * (scale - 1)) * 0.5)
        scaled_bboxes = (
            bboxes + np.stack(((- dw), (- dh), dw, dh), axis=(- 1)))
    if (clip_shape is not None):
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def imcrop(img, bboxes, scale=1.0, pad_fill=None):
    'Crop image patches.\n\n    3 steps: scale the bboxes -> clip bboxes -> crop and pad.\n\n    Args:\n        img (ndarray): Image to be cropped.\n        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.\n        scale (float, optional): Scale ratio of bboxes, the default value\n            1.0 means no padding.\n        pad_fill (Number | list[Number]): Value to be filled for padding.\n            Default: None, which means no padding.\n\n    Returns:\n        list[ndarray] | ndarray: The cropped image patches.\n    '
    chn = (1 if (img.ndim == 2) else img.shape[2])
    if (pad_fill is not None):
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert (len(pad_fill) == chn)
    _bboxes = (bboxes[(None, ...)] if (bboxes.ndim == 1) else bboxes)
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)
    patches = []
    for i in range(clipped_bbox.shape[0]):
        (x1, y1, x2, y2) = tuple(clipped_bbox[i, :])
        if (pad_fill is None):
            patch = img[y1:(y2 + 1), x1:(x2 + 1), ...]
        else:
            (_x1, _y1, _x2, _y2) = tuple(scaled_bboxes[i, :])
            if (chn == 1):
                patch_shape = (((_y2 - _y1) + 1), ((_x2 - _x1) + 1))
            else:
                patch_shape = (((_y2 - _y1) + 1), ((_x2 - _x1) + 1), chn)
            patch = (np.array(pad_fill, dtype=img.dtype) *
                     np.ones(patch_shape, dtype=img.dtype))
            x_start = (0 if (_x1 >= 0) else (- _x1))
            y_start = (0 if (_y1 >= 0) else (- _y1))
            w = ((x2 - x1) + 1)
            h = ((y2 - y1) + 1)
            patch[y_start:(y_start + h), x_start:(x_start +
                                                  w), ...] = img[y1:(y1 + h), x1:(x1 + w), ...]
        patches.append(patch)
    if (bboxes.ndim == 1):
        return patches[0]
    else:
        return patches


def impad(img, *, shape=None, padding=None, pad_val=0, padding_mode='constant'):
    "Pad the given image to a certain shape or pad on all sides with\n    specified padding mode and padding value.\n\n    Args:\n        img (ndarray): Image to be padded.\n        shape (tuple[int]): Expected padding shape (h, w). Default: None.\n        padding (int or tuple[int]): Padding on each border. If a single int is\n            provided this is used to pad all borders. If tuple of length 2 is\n            provided this is the padding on left/right and top/bottom\n            respectively. If a tuple of length 4 is provided this is the\n            padding for the left, top, right and bottom borders respectively.\n            Default: None. Note that `shape` and `padding` can not be both\n            set.\n        pad_val (Number | Sequence[Number]): Values to be filled in padding\n            areas when padding_mode is 'constant'. Default: 0.\n        padding_mode (str): Type of padding. Should be: constant, edge,\n            reflect or symmetric. Default: constant.\n\n            - constant: pads with a constant value, this value is specified\n                with pad_val.\n            - edge: pads with the last value at the edge of the image.\n            - reflect: pads with reflection of image without repeating the\n                last value on the edge. For example, padding [1, 2, 3, 4]\n                with 2 elements on both sides in reflect mode will result\n                in [3, 2, 1, 2, 3, 4, 3, 2].\n            - symmetric: pads with reflection of image repeating the last\n                value on the edge. For example, padding [1, 2, 3, 4] with\n                2 elements on both sides in symmetric mode will result in\n                [2, 1, 1, 2, 3, 4, 4, 3]\n\n    Returns:\n        ndarray: The padded image.\n    "
    assert ((shape is not None) ^ (padding is not None))
    if (shape is not None):
        padding = (0, 0, (shape[1] - img.shape[1]), (shape[0] - img.shape[0]))
    if isinstance(pad_val, tuple):
        assert (len(pad_val) == img.shape[(- 1)])
    elif (not isinstance(pad_val, numbers.Number)):
        raise TypeError(''.join(
            ['pad_val must be a int or a tuple. But received ', '{}'.format(type(pad_val))]))
    if (isinstance(padding, tuple) and (len(padding) in [2, 4])):
        if (len(padding) == 2):
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(''.join(
            ['Padding must be a int or a 2, or 4 element tuple.But received ', '{}'.format(padding)]))
    assert (padding_mode in ['constant', 'edge', 'reflect', 'symmetric'])
    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT,
    }
    img = cv2.copyMakeBorder(
        img, padding[1], padding[3], padding[0], padding[2], border_type[padding_mode], value=pad_val)
    return img


def impad_to_multiple(img, divisor, pad_val=0):
    'Pad an image to ensure each edge to be multiple to some number.\n\n    Args:\n        img (ndarray): Image to be padded.\n        divisor (int): Padded image edges will be multiple to divisor.\n        pad_val (Number | Sequence[Number]): Same as :func:`impad`.\n\n    Returns:\n        ndarray: The padded image.\n    '
    pad_h = (int(np.ceil((img.shape[0] / divisor))) * divisor)
    pad_w = (int(np.ceil((img.shape[1] / divisor))) * divisor)
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


def _get_shear_matrix(magnitude, direction='horizontal'):
    'Generate the shear matrix for transformation.\n\n    Args:\n        magnitude (int | float): The magnitude used for shear.\n        direction (str): Thie flip direction, either "horizontal"\n            or "vertical".\n\n    Returns:\n        ndarray: The shear matrix with dtype float32.\n    '
    if (direction == 'horizontal'):
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif (direction == 'vertical'):
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix


def imshear(img, magnitude, direction='horizontal', border_value=0, interpolation='bilinear'):
    'Shear an image.\n\n    Args:\n        img (ndarray): Image to be sheared with format (h, w)\n            or (h, w, c).\n        magnitude (int | float): The magnitude used for shear.\n        direction (str): Thie flip direction, either "horizontal"\n            or "vertical".\n        border_value (int | tuple[int]): Value used in case of a\n            constant border.\n        interpolation (str): Same as :func:`resize`.\n\n    Returns:\n        ndarray: The sheared image.\n    '
    assert (direction in ['horizontal', 'vertical']), ''.join(
        ['Invalid direction: ', '{}'.format(direction)])
    (height, width) = img.shape[:2]
    if (img.ndim == 2):
        channels = 1
    elif (img.ndim == 3):
        channels = img.shape[(- 1)]
    if isinstance(border_value, int):
        border_value = tuple(([border_value] * channels))
    elif isinstance(border_value, tuple):
        assert (len(border_value) == channels), 'Expected the num of elements in tuple equals the channelsof input image. Found {} vs {}'.format(
            len(border_value), channels)
    else:
        raise ValueError(''.join(['Invalid type ', '{}'.format(
            type(border_value)), ' for `border_value`']))
    shear_matrix = _get_shear_matrix(magnitude, direction)
    sheared = cv2.warpAffine(img, shear_matrix, (width, height),
                             borderValue=border_value[:3], flags=cv2_interp_codes[interpolation])
    return sheared


def _get_translate_matrix(offset, direction='horizontal'):
    'Generate the translate matrix.\n\n    Args:\n        offset (int | float): The offset used for translate.\n        direction (str): The translate direction, either\n            "horizontal" or "vertical".\n\n    Returns:\n        ndarray: The translate matrix with dtype float32.\n    '
    if (direction == 'horizontal'):
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif (direction == 'vertical'):
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def imtranslate(img, offset, direction='horizontal', border_value=0, interpolation='bilinear'):
    'Translate an image.\n\n    Args:\n        img (ndarray): Image to be translated with format\n            (h, w) or (h, w, c).\n        offset (int | float): The offset used for translate.\n        direction (str): The translate direction, either "horizontal"\n            or "vertical".\n        border_value (int | tuple[int]): Value used in case of a\n            constant border.\n        interpolation (str): Same as :func:`resize`.\n\n    Returns:\n        ndarray: The translated image.\n    '
    assert (direction in ['horizontal', 'vertical']), ''.join(
        ['Invalid direction: ', '{}'.format(direction)])
    (height, width) = img.shape[:2]
    if (img.ndim == 2):
        channels = 1
    elif (img.ndim == 3):
        channels = img.shape[(- 1)]
    if isinstance(border_value, int):
        border_value = tuple(([border_value] * channels))
    elif isinstance(border_value, tuple):
        assert (len(border_value) == channels), 'Expected the num of elements in tuple equals the channelsof input image. Found {} vs {}'.format(
            len(border_value), channels)
    else:
        raise ValueError(''.join(['Invalid type ', '{}'.format(
            type(border_value)), ' for `border_value`.']))
    translate_matrix = _get_translate_matrix(offset, direction)
    translated = cv2.warpAffine(img, translate_matrix, (width, height),
                                borderValue=border_value[:3], flags=cv2_interp_codes[interpolation])
    return translated
