
import inspect
import warnings
from functools import partial
from .misc import is_str


class Registry():
    'A registry to map strings to classes.\n\n    Args:\n        name (str): Registry name.\n    '

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return (self.get(key) is not None)

    def __repr__(self):
        format_str = (self.__class__.__name__ + ''.join(['(name=', '{}'.format(
            self._name), ', items=', '{}'.format(self._module_dict), ')']))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        'Get the registry record.\n\n        Args:\n            key (str): The class name in string format.\n\n        Returns:\n            class: The corresponding class.\n        '
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if (not inspect.isclass(module_class)):
            raise TypeError(
                ''.join(['module must be a class, but got ', '{}'.format(type(module_class))]))
        if (module_name is None):
            module_name = module_class.__name__
        if ((not force) and (module_name in self._module_dict)):
            raise KeyError(''.join(
                ['{}'.format(module_name), ' is already registered in ', '{}'.format(self.name)]))
        self._module_dict[module_name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn('The old API of register_module(module, force=False) is deprecated and will be removed, please use the new API register_module(name=None, force=False, module=None) instead.')
        if (cls is None):
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        "Register a module.\n\n        A record will be added to `self._module_dict`, whose key is the class\n        name or the specified name, and value is the class itself.\n        It can be used as a decorator or a normal function.\n\n        Example:\n            >>> backbones = Registry('backbone')\n            >>> @backbones.register_module()\n            >>> class ResNet:\n            >>>     pass\n\n            >>> backbones = Registry('backbone')\n            >>> @backbones.register_module(name='mnet')\n            >>> class MobileNet:\n            >>>     pass\n\n            >>> backbones = Registry('backbone')\n            >>> class ResNet:\n            >>>     pass\n            >>> backbones.register_module(ResNet)\n\n        Args:\n            name (str | None): The module name to be registered. If not\n                specified, the class name will be used.\n            force (bool, optional): Whether to override an existing class with\n                the same name. Default: False.\n            module (type): Module class to be registered.\n        "
        if (not isinstance(force, bool)):
            raise TypeError(
                ''.join(['force must be a boolean, but got ', '{}'.format(type(force))]))
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)
        if (module is not None):
            self._register_module(module_class=module,
                                  module_name=name, force=force)
            return module
        if (not ((name is None) or isinstance(name, str))):
            raise TypeError(
                ''.join(['name must be a str, but got ', '{}'.format(type(name))]))

        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls
        return _register


def build_from_cfg(cfg, registry, default_args=None):
    'Build a module from config dict.\n\n    Args:\n        cfg (dict): Config dict. It should at least contain the key "type".\n        registry (:obj:`Registry`): The registry to search the type from.\n        default_args (dict, optional): Default initialization arguments.\n\n    Returns:\n        object: The constructed object.\n    '
    if (not isinstance(cfg, dict)):
        raise TypeError(
            ''.join(['cfg must be a dict, but got ', '{}'.format(type(cfg))]))
    if ('type' not in cfg):
        if ((default_args is None) or ('type' not in default_args)):
            raise KeyError(''.join(['`cfg` or `default_args` must contain the key "type", but got ', '{}'.format(
                cfg), '\n', '{}'.format(default_args)]))
    if (not isinstance(registry, Registry)):
        raise TypeError(''.join(
            ['registry must be an mmcv.Registry object, but got ', '{}'.format(type(registry))]))
    if (not (isinstance(default_args, dict) or (default_args is None))):
        raise TypeError(''.join(
            ['default_args must be a dict or None, but got ', '{}'.format(type(default_args))]))
    args = cfg.copy()
    if (default_args is not None):
        for (name, value) in default_args.items():
            args.setdefault(name, value)
    obj_type = args.pop('type')
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if (obj_cls is None):
            raise KeyError(''.join(['{}'.format(
                obj_type), ' is not in the ', '{}'.format(registry.name), ' registry']))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(''.join(
            ['type must be a str or valid type, but got ', '{}'.format(type(obj_type))]))
    return obj_cls(**args)
