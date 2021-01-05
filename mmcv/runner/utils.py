
import os
import random
import sys
import time
from getpass import getuser
from socket import gethostname
import numpy as np
import torch
import mmcv


def get_host_info():
    return ''.join(['{}'.format(getuser()), '@', '{}'.format(gethostname())])


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def obj_from_dict(info, parent=None, default_args=None):
    'Initialize an object from dict.\n\n    The dict must contain the key "type", which indicates the object type, it\n    can be either a string or type, such as "list" or ``list``. Remaining\n    fields are treated as the arguments for constructing the object.\n\n    Args:\n        info (dict): Object types and arguments.\n        parent (:class:`module`): Module which may containing expected object\n            classes.\n        default_args (dict, optional): Default arguments for initializing the\n            object.\n\n    Returns:\n        any type: Object built from the dict.\n    '
    assert (isinstance(info, dict) and ('type' in info))
    assert (isinstance(default_args, dict) or (default_args is None))
    args = info.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        if (parent is not None):
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif (not isinstance(obj_type, type)):
        raise TypeError(''.join(
            ['type must be a str or valid type, but got ', '{}'.format(type(obj_type))]))
    if (default_args is not None):
        for (name, value) in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    'Set random seed.\n\n    Args:\n        seed (int): Seed to be used.\n        deterministic (bool): Whether to set the deterministic option for\n            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`\n            to True and `torch.backends.cudnn.benchmark` to False.\n            Default: False.\n        rank_shift (bool): Whether to add rank number to the random seed to\n            have different random seed in different threads. Default: False.\n    '
    if use_rank_shift:
        (rank, _) = mmcv.runner.get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
