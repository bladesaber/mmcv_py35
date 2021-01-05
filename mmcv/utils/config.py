
import ast
import os.path as osp
import platform
import shutil
import sys
import tempfile
from argparse import Action, ArgumentParser
from collections import abc
from importlib import import_module
from addict import Dict
from yapf.yapflib.yapf_api import FormatCode
from .misc import import_modules_from_strings
from .path import check_file_exist
if (platform.system() == 'Windows'):
    import regex as re
else:
    import re
BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                "'{self.__class__.__name__}' object has no attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for (k, v) in cfg.items():
        if isinstance(v, str):
            parser.add_argument((('--' + prefix) + k))
        elif isinstance(v, int):
            parser.add_argument((('--' + prefix) + k), type=int)
        elif isinstance(v, float):
            parser.add_argument((('--' + prefix) + k), type=float)
        elif isinstance(v, bool):
            parser.add_argument((('--' + prefix) + k), action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, ((prefix + k) + '.'))
        elif isinstance(v, abc.Iterable):
            parser.add_argument((('--' + prefix) + k),
                                type=type(v[0]), nargs='+')
        else:
            print(''.join(['cannot parse key ', '{}'.format(
                (prefix + k)), ' of type ', '{}'.format(type(v))]))
    return parser


class Config():
    'A facility for config and config files.\n\n    It supports common file formats as configs: python/json/yaml. The interface\n    is the same as a dict object and also allows access config values as\n    attributes.\n\n    Example:\n        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))\n        >>> cfg.a\n        1\n        >>> cfg.b\n        {\'b1\': [0, 1]}\n        >>> cfg.b.b1\n        [0, 1]\n        >>> cfg = Config.fromfile(\'tests/data/config/a.py\')\n        >>> cfg.filename\n        "/home/kchen/projects/mmcv/tests/data/config/a.py"\n        >>> cfg.item4\n        \'test\'\n        >>> cfg\n        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "\n        "{\'item1\': [1, 2], \'item2\': {\'a\': 0}, \'item3\': True, \'item4\': \'test\'}"\n    '

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(''.join(['There are syntax errors in config file ', '{}'.format(
                filename), ': ', '{}'.format(e)]))

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(fileDirname=file_dirname, fileBasename=file_basename,
                                 fileBasenameNoExtension=file_basename_no_extension, fileExtname=file_extname)
        with open(filename, 'r') as f:
            config_file = f.read()
        for (key, value) in support_templates.items():
            regexp = (('\\{\\{\\s*' + str(key)) + '\\s*\\}\\}')
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if (fileExtname not in ['.py', '.json', '.yaml', '.yml']):
            raise IOError('Only py/yml/yaml/json type are supported now!')
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if (platform.system() == 'Windows'):
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            if use_predefined_variables:
                Config._substitute_predefined_vars(
                    filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {name: value for (name, value) in mod.__dict__.items() if (
                    not name.startswith('__'))}
                del sys.modules[temp_module_name]
            elif filename.endswith(('.yml', '.yaml', '.json')):
                import mmcv
                cfg_dict = mmcv.load(temp_config_file.name)
            temp_config_file.close()
        cfg_text = (filename + '\n')
        with open(filename, 'r') as f:
            cfg_text += f.read()
        if (BASE_KEY in cfg_dict):
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = (base_filename if isinstance(
                base_filename, list) else [base_filename])
            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                (_cfg_dict, _cfg_text) = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)
            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if (len((base_cfg_dict.keys() & c.keys())) > 0):
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)
            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)
        return (cfg_dict, cfg_text)

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        "merge dict ``a`` into dict ``b`` (non-inplace).\n\n        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid\n        in-place modifications.\n\n        Args:\n            a (dict): The source dict to be merged into ``b``.\n            b (dict): The origin dict to be fetch keys from ``a``.\n            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')\n              are allowed in source ``a`` and will replace the element of the\n              corresponding index in b if b is a list. Default: False.\n\n        Returns:\n            dict: The modified dict of ``b`` using ``a``.\n\n        Examples:\n            # Normally merge a into b.\n            >>> Config._merge_a_into_b(\n            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))\n            {'obj': {'a': 2}}\n\n            # Delete b first and merge a into b.\n            >>> Config._merge_a_into_b(\n            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))\n            {'obj': {'a': 2}}\n\n            # b is a list\n            >>> Config._merge_a_into_b(\n            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)\n            [{'a': 2}, {'b': 2}]\n        "
        b = b.copy()
        for (k, v) in a.items():
            if (allow_list_keys and k.isdigit() and isinstance(b, list)):
                k = int(k)
                if (len(b) <= k):
                    raise KeyError(''.join(['Index ', '{}'.format(
                        k), ' exceeds the length of list ', '{}'.format(b)]))
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif (isinstance(v, dict) and (k in b) and (not v.pop(DELETE_KEY, False))):
                allowed_types = ((dict, list) if allow_list_keys else dict)
                if (not isinstance(b[k], allowed_types)):
                    raise TypeError(''.join(['{}'.format(k), '=', '{}'.format(v), ' in child config cannot inherit from base because ', '{}'.format(
                        k), ' is a dict in the child config but is of type ', '{}'.format(type(b[k])), ' in base config. You may set `', '{}'.format(DELETE_KEY), '=True` to ignore the base config']))
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename, use_predefined_variables=True, import_custom_modules=True):
        (cfg_dict, cfg_text) = Config._file2dict(
            filename, use_predefined_variables)
        if (import_custom_modules and cfg_dict.get('custom_imports', None)):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        'Generate argparser from config file automatically (experimental)'
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return (parser, cfg)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if (cfg_dict is None):
            cfg_dict = dict()
        elif (not isinstance(cfg_dict, dict)):
            raise TypeError(
                ''.join(['cfg_dict must be a dict, but got ', '{}'.format(type(cfg_dict))]))
        for key in cfg_dict:
            if (key in RESERVED_KEYS):
                raise KeyError(
                    ''.join(['{}'.format(key), ' is reserved for config file']))
        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):
        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if (len(s) == 1):
                return s_
            first = s.pop(0)
            s = [((num_spaces * ' ') + line) for line in s]
            s = '\n'.join(s)
            s = ((first + '\n') + s)
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = ''.join(["'", '{}'.format(v), "'"])
            else:
                v_str = str(v)
            if use_mapping:
                k_str = (''.join(["'", '{}'.format(k), "'"])
                         if isinstance(k, str) else str(k))
                attr_str = ''.join(
                    ['{}'.format(k_str), ': ', '{}'.format(v_str)])
            else:
                attr_str = ''.join(
                    ['{}'.format(str(k)), '=', '{}'.format(v_str)])
            attr_str = _indent(attr_str, indent)
            return attr_str

        def _format_list(k, v, use_mapping=False):
            if all((isinstance(_, dict) for _ in v)):
                v_str = '[\n'
                v_str += '\n'.join((''.join(['dict(', '{}'.format(
                    _indent(_format_dict(v_), indent)), '),']) for v_ in v)).rstrip(',')
                if use_mapping:
                    k_str = (''.join(["'", '{}'.format(k), "'"])
                             if isinstance(k, str) else str(k))
                    attr_str = ''.join(
                        ['{}'.format(k_str), ': ', '{}'.format(v_str)])
                else:
                    attr_str = ''.join(
                        ['{}'.format(str(k)), '=', '{}'.format(v_str)])
                attr_str = (_indent(attr_str, indent) + ']')
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= (
                    not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []
            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for (idx, (k, v)) in enumerate(input_dict.items()):
                is_last = (idx >= (len(input_dict) - 1))
                end = ('' if (outest_level or is_last) else ',')
                if isinstance(v, dict):
                    v_str = ('\n' + _format_dict(v))
                    if use_mapping:
                        k_str = (''.join(["'", '{}'.format(k), "'"])
                                 if isinstance(k, str) else str(k))
                        attr_str = ''.join(
                            ['{}'.format(k_str), ': dict(', '{}'.format(v_str)])
                    else:
                        attr_str = ''.join(
                            ['{}'.format(str(k)), '=dict(', '{}'.format(v_str)])
                    attr_str = ((_indent(attr_str, indent) + ')') + end)
                elif isinstance(v, list):
                    attr_str = (_format_list(k, v, use_mapping) + end)
                else:
                    attr_str = (_format_basic_types(k, v, use_mapping) + end)
                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r
        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        yapf_style = dict(based_on_style='pep8', blank_line_before_nested_class_or_def=True,
                          split_before_expression_after_opening_paren=True)
        (text, _) = FormatCode(text, style_config=yapf_style, verify=True)
        return text

    def __repr__(self):
        return ''.join(['Config (path: ', '{}'.format(self.filename), '): ', '{}'.format(self._cfg_dict.__repr__())])

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __setstate__(self, state):
        (_cfg_dict, _filename, _text) = state
        super(Config, self).__setattr__('_cfg_dict', _cfg_dict)
        super(Config, self).__setattr__('_filename', _filename)
        super(Config, self).__setattr__('_text', _text)

    def dump(self, file=None):
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict').to_dict()
        if self.filename.endswith('.py'):
            if (file is None):
                return self.pretty_text
            else:
                with open(file, 'w') as f:
                    f.write(self.pretty_text)
        else:
            import mmcv
            if (file is None):
                file_format = self.filename.split('.')[(- 1)]
                return mmcv.dump(cfg_dict, file_format=file_format)
            else:
                mmcv.dump(cfg_dict, file)

    def merge_from_dict(self, options, allow_list_keys=True):
        "Merge list into cfg_dict.\n\n        Merge the dict parsed by MultipleKVAction into this cfg.\n\n        Examples:\n            >>> options = {'model.backbone.depth': 50,\n            ...            'model.backbone.with_cp':True}\n            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))\n            >>> cfg.merge_from_dict(options)\n            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')\n            >>> assert cfg_dict == dict(\n            ...     model=dict(backbone=dict(depth=50, with_cp=True)))\n\n            # Merge list element\n            >>> cfg = Config(dict(pipeline=[\n            ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))\n            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})\n            >>> cfg.merge_from_dict(options, allow_list_keys=True)\n            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')\n            >>> assert cfg_dict == dict(pipeline=[\n            ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])\n\n        Args:\n            options (dict): dict of configs to merge from.\n            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')\n              are allowed in ``options`` and will replace the element of the\n              corresponding index in the config if the config is a list.\n              Default: True.\n        "
        option_cfg_dict = {

        }
        for (full_key, v) in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:(- 1)]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[(- 1)]
            d[subkey] = v
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        super(Config, self).__setattr__('_cfg_dict', Config._merge_a_into_b(
            option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys))


class DictAction(Action):
    "\n    argparse action to split an argument into KEY=VALUE form\n    on the first = and append to a dictionary. List options can\n    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit\n    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build\n    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'\n    "

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if (val.lower() in ['true', 'false']):
            return (True if (val.lower() == 'true') else False)
        return val

    @staticmethod
    def _parse_iterable(val):
        "Parse iterable values in the string.\n\n        All elements inside '()' or '[]' are treated as iterable values.\n\n        Args:\n            val (str): Value string.\n\n        Returns:\n            list | tuple: The expanded list or tuple from the string.\n\n        Examples:\n            >>> DictAction._parse_iterable('1,2,3')\n            [1, 2, 3]\n            >>> DictAction._parse_iterable('[a, b, c]')\n            ['a', 'b', 'c']\n            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')\n            [(1, 2, 3), ['a', 'b], 'c']\n        "

        def find_next_comma(string):
            "Find the position of next comma in the string.\n\n            If no ',' is found in the string, return the string length. All\n            chars inside '()' and '[]' are treated as one element and thus ','\n            inside these brackets are ignored.\n            "
            assert ((string.count('(') == string.count(')')) and (string.count('[') == string.count(
                ']'))), ''.join(['Imbalanced brackets exist in ', '{}'.format(string)])
            end = len(string)
            for (idx, char) in enumerate(string):
                pre = string[:idx]
                if ((char == ',') and (pre.count('(') == pre.count(')')) and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end
        val = val.strip('\'"').replace(' ', '')
        is_tuple = False
        if (val.startswith('(') and val.endswith(')')):
            is_tuple = True
            val = val[1:(- 1)]
        elif (val.startswith('[') and val.endswith(']')):
            val = val[1:(- 1)]
        elif (',' not in val):
            return DictAction._parse_int_float_bool(val)
        values = []
        while (len(val) > 0):
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[(comma_idx + 1):]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {

        }
        for kv in values:
            (key, val) = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
