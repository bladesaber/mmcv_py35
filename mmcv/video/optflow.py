
import warnings
import numpy as np
from mmcv.arraymisc import dequantize, quantize
from mmcv.image import imread, imwrite
from mmcv.utils import is_str


def flowread(flow_or_path, quantize=False, concat_axis=0, *args, **kwargs):
    'Read an optical flow map.\n\n    Args:\n        flow_or_path (ndarray or str): A flow map or filepath.\n        quantize (bool): whether to read quantized pair, if set to True,\n            remaining args will be passed to :func:`dequantize_flow`.\n        concat_axis (int): The axis that dx and dy are concatenated,\n            can be either 0 or 1. Ignored if quantize is False.\n\n    Returns:\n        ndarray: Optical flow represented as a (h, w, 2) numpy array\n    '
    if isinstance(flow_or_path, np.ndarray):
        if ((flow_or_path.ndim != 3) or (flow_or_path.shape[(- 1)] != 2)):
            raise ValueError(
                ''.join(['Invalid flow with shape ', '{}'.format(flow_or_path.shape)]))
        return flow_or_path
    elif (not is_str(flow_or_path)):
        raise TypeError(''.join(
            ['"flow_or_path" must be a filename or numpy array, not ', '{}'.format(type(flow_or_path))]))
    if (not quantize):
        with open(flow_or_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except Exception:
                raise IOError(
                    ''.join(['Invalid flow file: ', '{}'.format(flow_or_path)]))
            else:
                if (header != 'PIEH'):
                    raise IOError(''.join(['Invalid flow file: ', '{}'.format(
                        flow_or_path), ', header does not contain PIEH']))
            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, ((w * h) * 2)).reshape((h, w, 2))
    else:
        assert (concat_axis in [0, 1])
        cat_flow = imread(flow_or_path, flag='unchanged')
        if (cat_flow.ndim != 2):
            raise IOError(''.join(['{}'.format(
                flow_or_path), ' is not a valid quantized flow file, its dimension is ', '{}'.format(cat_flow.ndim), '.']))
        assert ((cat_flow.shape[concat_axis] % 2) == 0)
        (dx, dy) = np.split(cat_flow, 2, axis=concat_axis)
        flow = dequantize_flow(dx, dy, *args, **kwargs)
    return flow.astype(np.float32)


def flowwrite(flow, filename, quantize=False, concat_axis=0, *args, **kwargs):
    'Write optical flow to file.\n\n    If the flow is not quantized, it will be saved as a .flo file losslessly,\n    otherwise a jpeg image which is lossy but of much smaller size. (dx and dy\n    will be concatenated horizontally into a single image if quantize is True.)\n\n    Args:\n        flow (ndarray): (h, w, 2) array of optical flow.\n        filename (str): Output filepath.\n        quantize (bool): Whether to quantize the flow and save it to 2 jpeg\n            images. If set to True, remaining args will be passed to\n            :func:`quantize_flow`.\n        concat_axis (int): The axis that dx and dy are concatenated,\n            can be either 0 or 1. Ignored if quantize is False.\n    '
    if (not quantize):
        with open(filename, 'wb') as f:
            f.write('PIEH'.encode('utf-8'))
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)
            f.flush()
    else:
        assert (concat_axis in [0, 1])
        (dx, dy) = quantize_flow(flow, *args, **kwargs)
        dxdy = np.concatenate((dx, dy), axis=concat_axis)
        imwrite(dxdy, filename)


def quantize_flow(flow, max_val=0.02, norm=True):
    'Quantize flow to [0, 255].\n\n    After this step, the size of flow will be much smaller, and can be\n    dumped as jpeg images.\n\n    Args:\n        flow (ndarray): (h, w, 2) array of optical flow.\n        max_val (float): Maximum value of flow, values beyond\n                        [-max_val, max_val] will be truncated.\n        norm (bool): Whether to divide flow values by image width/height.\n\n    Returns:\n        tuple[ndarray]: Quantized dx and dy.\n    '
    (h, w, _) = flow.shape
    dx = flow[(..., 0)]
    dy = flow[(..., 1)]
    if norm:
        dx = (dx / w)
        dy = (dy / h)
    flow_comps = [quantize(d, (- max_val), max_val, 255, np.uint8)
                  for d in [dx, dy]]
    return tuple(flow_comps)


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
    'Recover from quantized flow.\n\n    Args:\n        dx (ndarray): Quantized dx.\n        dy (ndarray): Quantized dy.\n        max_val (float): Maximum value used when quantizing.\n        denorm (bool): Whether to multiply flow values with width/height.\n\n    Returns:\n        ndarray: Dequantized flow.\n    '
    assert (dx.shape == dy.shape)
    assert ((dx.ndim == 2) or ((dx.ndim == 3) and (dx.shape[(- 1)] == 1)))
    (dx, dy) = [dequantize(d, (- max_val), max_val, 255) for d in [dx, dy]]
    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow


def flow_warp(img, flow, filling_value=0, interpolate_mode='nearest'):
    'Use flow to warp img.\n\n    Args:\n        img (ndarray, float or uint8): Image to be warped.\n        flow (ndarray, float): Optical Flow.\n        filling_value (int): The missing pixels will be set with filling_value.\n        interpolate_mode (str): bilinear -> Bilinear Interpolation;\n                                nearest -> Nearest Neighbor.\n\n    Returns:\n        ndarray: Warped image with the same shape of img\n    '
    warnings.warn(
        'This function is just for prototyping and cannot guarantee the computational efficiency.')
    assert (flow.ndim == 3), 'Flow must be in 3D arrays.'
    height = flow.shape[0]
    width = flow.shape[1]
    channels = img.shape[2]
    output = (np.ones((height, width, channels),
                      dtype=img.dtype) * filling_value)
    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = (grid[:, :, 0] + flow[:, :, 1])
    dy = (grid[:, :, 1] + flow[:, :, 0])
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = ((((sx >= 0) & (sx < (height - 1)))
              & (sy >= 0)) & (sy < (width - 1)))
    if (interpolate_mode == 'nearest'):
        output[valid, :] = img[dx[valid].round().astype(
            int), dy[valid].round().astype(int), :]
    elif (interpolate_mode == 'bilinear'):
        eps_ = 1e-06
        (dx, dy) = ((dx + eps_), (dy + eps_))
        left_top_ = ((img[np.floor(dx[valid]).astype(int), np.floor(dy[valid]).astype(
            int), :] * (np.ceil(dx[valid]) - dx[valid])[:, None]) * (np.ceil(dy[valid]) - dy[valid])[:, None])
        left_down_ = ((img[np.ceil(dx[valid]).astype(int), np.floor(dy[valid]).astype(
            int), :] * (dx[valid] - np.floor(dx[valid]))[:, None]) * (np.ceil(dy[valid]) - dy[valid])[:, None])
        right_top_ = ((img[np.floor(dx[valid]).astype(int), np.ceil(dy[valid]).astype(
            int), :] * (np.ceil(dx[valid]) - dx[valid])[:, None]) * (dy[valid] - np.floor(dy[valid]))[:, None])
        right_down_ = ((img[np.ceil(dx[valid]).astype(int), np.ceil(dy[valid]).astype(
            int), :] * (dx[valid] - np.floor(dx[valid]))[:, None]) * (dy[valid] - np.floor(dy[valid]))[:, None])
        output[valid, :] = (
            ((left_top_ + left_down_) + right_top_) + right_down_)
    else:
        raise NotImplementedError(''.join(
            ['We only support interpolation modes of nearest and bilinear, but got ', '{}'.format(interpolate_mode), '.']))
    return output.astype(img.dtype)
