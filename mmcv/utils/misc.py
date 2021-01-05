
import functools
import itertools
import subprocess
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec


def is_str(x):
    'Whether the input is an string instance.\n\n    Note: This method is deprecated since python 2 is no longer supported.\n    '
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    "Import modules from the given list of strings.\n\n    Args:\n        imports (list | str | None): The given module names to be imported.\n        allow_failed_imports (bool): If True, the failed imports will return\n            None. Otherwise, an ImportError is raise. Default: False.\n\n    Returns:\n        list[module] | module | None: The imported modules.\n\n    Examples:\n        >>> osp, sys = import_modules_from_strings(\n        ...     ['os.path', 'sys'])\n        >>> import os.path as osp_\n        >>> import sys as sys_\n        >>> assert osp == osp_\n        >>> assert sys == sys_\n    "
    if (not imports):
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if (not isinstance(imports, list)):
        raise TypeError(''.join(
            ['custom_imports must be a list but got type ', '{}'.format(type(imports))]))
    imported = []
    for imp in imports:
        if (not isinstance(imp, str)):
            raise TypeError(''.join(['{}'.format(imp), ' is of type ', '{}'.format(
                type(imp)), ' and cannot be imported.']))
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(
                    ''.join(['{}'.format(imp), ' failed to import and is ignored.']), UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def iter_cast(inputs, dst_type, return_type=None):
    'Cast elements of an iterable object into some type.\n\n    Args:\n        inputs (Iterable): The input object.\n        dst_type (type): Destination type.\n        return_type (type, optional): If specified, the output object will be\n            converted to this type, otherwise an iterator.\n\n    Returns:\n        iterator or specified type: The converted object.\n    '
    if (not isinstance(inputs, abc.Iterable)):
        raise TypeError('inputs must be an iterable object')
    if (not isinstance(dst_type, type)):
        raise TypeError('"dst_type" must be a valid type')
    out_iterable = map(dst_type, inputs)
    if (return_type is None):
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    'Cast elements of an iterable object into a list of some type.\n\n    A partial method of :func:`iter_cast`.\n    '
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    'Cast elements of an iterable object into a tuple of some type.\n\n    A partial method of :func:`iter_cast`.\n    '
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    'Check whether it is a sequence of some type.\n\n    Args:\n        seq (Sequence): The sequence to be checked.\n        expected_type (type): Expected type of sequence items.\n        seq_type (type, optional): Expected sequence type.\n\n    Returns:\n        bool: Whether the sequence is valid.\n    '
    if (seq_type is None):
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if (not isinstance(seq, exp_seq_type)):
        return False
    for item in seq:
        if (not isinstance(item, expected_type)):
            return False
    return True


def is_list_of(seq, expected_type):
    'Check whether it is a list of some type.\n\n    A partial method of :func:`is_seq_of`.\n    '
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    'Check whether it is a tuple of some type.\n\n    A partial method of :func:`is_seq_of`.\n    '
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    'Slice a list into several sub lists by a list of given length.\n\n    Args:\n        in_list (list): The list to be sliced.\n        lens(int or list): The expected length of each out list.\n\n    Returns:\n        list: A list of sliced list.\n    '
    if isinstance(lens, int):
        assert ((len(in_list) % lens) == 0)
        lens = ([lens] * int((len(in_list) / lens)))
    if (not isinstance(lens, list)):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif (sum(lens) != len(in_list)):
        raise ValueError(''.join(['sum of lens and list length does not match: ', '{}'.format(
            sum(lens)), ' != ', '{}'.format(len(in_list))]))
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:(idx + lens[i])])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    'Concatenate a list of list into a single list.\n\n    Args:\n        in_list (list): The list of list to be merged.\n\n    Returns:\n        list: The concatenated flat list.\n    '
    return list(itertools.chain(*in_list))


def check_prerequisites(prerequisites, checker, msg_tmpl='Prerequisites "{}" are required in method "{}" but not found, please install them first.'):
    'A decorator factory to check if prerequisites are satisfied.\n\n    Args:\n        prerequisites (str of list[str]): Prerequisites to be checked.\n        checker (callable): The checker method that returns True if a\n            prerequisite is meet, False otherwise.\n        msg_tmpl (str): The message template with two variables.\n\n    Returns:\n        decorator: A specific decorator.\n    '

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = ([prerequisites] if isinstance(
                prerequisites, str) else prerequisites)
            missing = []
            for item in requirements:
                if (not checker(item)):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)
        return wrapped_func
    return wrap


def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    if (subprocess.call(''.join(['which ', '{}'.format(cmd)]), shell=True) != 0):
        return False
    else:
        return True


def requires_package(prerequisites):
    "A decorator to check if some python packages are installed.\n\n    Example:\n        >>> @requires_package('numpy')\n        >>> func(arg1, args):\n        >>>     return numpy.zeros(1)\n        array([0.])\n        >>> @requires_package(['numpy', 'non_package'])\n        >>> func(arg1, args):\n        >>>     return numpy.zeros(1)\n        ImportError\n    "
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    "A decorator to check if some executable files are installed.\n\n    Example:\n        >>> @requires_executable('ffmpeg')\n        >>> func(arg1, args):\n        >>>     print(1)\n        1\n    "
    return check_prerequisites(prerequisites, checker=_check_executable)


def deprecated_api_warning(name_dict, cls_name=None):
    'A decorator to check if some argments are deprecate and try to replace\n    deprecate src_arg_name to dst_arg_name.\n\n    Args:\n        name_dict(dict):\n            key (str): Deprecate argument names.\n            val (str): Expected argument names.\n\n    Returns:\n        func: New function.\n    '

    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            func_name = old_func.__name__
            if (cls_name is not None):
                func_name = ''.join(
                    ['{}'.format(cls_name), '.', '{}'.format(func_name)])
            if args:
                arg_names = args_info.args[:len(args)]
                for (src_arg_name, dst_arg_name) in name_dict.items():
                    if (src_arg_name in arg_names):
                        warnings.warn(''.join(['"', '{}'.format(src_arg_name), '" is deprecated in `', '{}'.format(
                            func_name), '`, please use "', '{}'.format(dst_arg_name), '" instead']))
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for (src_arg_name, dst_arg_name) in name_dict.items():
                    if (src_arg_name in kwargs):
                        warnings.warn(''.join(['"', '{}'.format(src_arg_name), '" is deprecated in `', '{}'.format(
                            func_name), '`, please use "', '{}'.format(dst_arg_name), '" instead']))
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)
            output = old_func(*args, **kwargs)
            return output
        return new_func
    return api_warning_wrapper
