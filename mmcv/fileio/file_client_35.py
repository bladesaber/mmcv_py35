
import inspect
from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    'Abstract class of storage backends.\n\n    All backends need to implement two apis: ``get()`` and ``get_text()``.\n    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file\n    as texts.\n    '

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class CephBackend(BaseStorageBackend):
    "Ceph storage backend.\n\n    Args:\n        path_mapping (dict|None): path mapping dict from local path to Petrel\n            path. When ``path_mapping={'src': 'dst'}``, ``src`` in ``filepath``\n            will be replaced by ``dst``. Default: None.\n    "

    def __init__(self, path_mapping=None):
        try:
            import ceph
        except ImportError:
            raise ImportError('Please install ceph to enable CephBackend.')
        self._client = ceph.S3Client()
        assert (isinstance(path_mapping, dict) or (path_mapping is None))
        self.path_mapping = path_mapping

    def get(self, filepath):
        filepath = str(filepath)
        if (self.path_mapping is not None):
            for (k, v) in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class PetrelBackend(BaseStorageBackend):
    "Petrel storage backend (for internal use).\n\n    Args:\n        path_mapping (dict|None): path mapping dict from local path to Petrel\n            path. When `path_mapping={'src': 'dst'}`, `src` in `filepath` will\n            be replaced by `dst`. Default: None.\n        enable_mc (bool): whether to enable memcached support. Default: True.\n    "

    def __init__(self, path_mapping=None, enable_mc=True):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError(
                'Please install petrel_client to enable PetrelBackend.')
        self._client = client.Client(enable_mc=enable_mc)
        assert (isinstance(path_mapping, dict) or (path_mapping is None))
        self.path_mapping = path_mapping

    def get(self, filepath):
        filepath = str(filepath)
        if (self.path_mapping is not None):
            for (k, v) in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class MemcachedBackend(BaseStorageBackend):
    'Memcached storage backend.\n\n    Attributes:\n        server_list_cfg (str): Config file for memcached server list.\n        client_cfg (str): Config file for memcached client.\n        sys_path (str | None): Additional path to be appended to `sys.path`.\n            Default: None.\n    '

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if (sys_path is not None):
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError(
                'Please install memcached to enable MemcachedBackend.')
        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(
            self.server_list_cfg, self.client_cfg)
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class LmdbBackend(BaseStorageBackend):
    'Lmdb storage backend.\n\n    Args:\n        db_path (str): Lmdb database path.\n        readonly (bool, optional): Lmdb environment parameter. If True,\n            disallow any write operations. Default: True.\n        lock (bool, optional): Lmdb environment parameter. If False, when\n            concurrent access occurs, do not lock the database. Default: False.\n        readahead (bool, optional): Lmdb environment parameter. If False,\n            disable the OS filesystem readahead mechanism, which may improve\n            random read performance when a database is larger than RAM.\n            Default: False.\n\n    Attributes:\n        db_path (str): Lmdb database path.\n    '

    def __init__(self, db_path, readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')
        self.db_path = str(db_path)
        self._client = lmdb.open(
            self.db_path, readonly=readonly, lock=lock, readahead=readahead, **kwargs)

    def get(self, filepath):
        'Get values according to the filepath.\n\n        Args:\n            filepath (str | obj:`Path`): Here, filepath is the lmdb key.\n        '
        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    'Raw hard disks storage backend.'

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf


class FileClient():
    'A general file client to access files in different backend.\n\n    The client loads a file or text in a specified backend from its path\n    and return it as a binary file. it can also register other backend\n    accessor with a given name and backend class.\n\n    Attributes:\n        backend (str): The storage backend type. Options are "disk", "ceph",\n            "memcached" and "lmdb".\n        client (:obj:`BaseStorageBackend`): The backend object.\n    '
    _backends = {
        'disk': HardDiskBackend,
        'ceph': CephBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
        'petrel': PetrelBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if (backend not in self._backends):
            raise ValueError(''.join(['Backend ', '{}'.format(
                backend), ' is not supported. Currently supported ones are ', '{}'.format(list(self._backends.keys()))]))
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    @classmethod
    def _register_backend(cls, name, backend, force=False):
        if (not isinstance(name, str)):
            raise TypeError(''.join(
                ['the backend name should be a string, but got ', '{}'.format(type(name))]))
        if (not inspect.isclass(backend)):
            raise TypeError(
                ''.join(['backend should be a class but got ', '{}'.format(type(backend))]))
        if (not issubclass(backend, BaseStorageBackend)):
            raise TypeError(''.join(['backend ', '{}'.format(
                backend), ' is not a subclass of BaseStorageBackend']))
        if ((not force) and (name in cls._backends)):
            raise KeyError(''.join(['{}'.format(
                name), ' is already registered as a storage backend, add "force=True" if you want to override it']))
        cls._backends[name] = backend

    @classmethod
    def register_backend(cls, name, backend=None, force=False):
        "Register a backend to FileClient.\n\n        This method can be used as a normal class method or a decorator.\n\n        .. code-block:: python\n\n            class NewBackend(BaseStorageBackend):\n\n                def get(self, filepath):\n                    return filepath\n\n                def get_text(self, filepath):\n                    return filepath\n\n            FileClient.register_backend('new', NewBackend)\n\n        or\n\n        .. code-block:: python\n\n            @FileClient.register_backend('new')\n            class NewBackend(BaseStorageBackend):\n\n                def get(self, filepath):\n                    return filepath\n\n                def get_text(self, filepath):\n                    return filepath\n\n        Args:\n            name (str): The name of the registered backend.\n            backend (class, optional): The backend class to be registered,\n                which must be a subclass of :class:`BaseStorageBackend`.\n                When this method is used as a decorator, backend is None.\n                Defaults to None.\n            force (bool, optional): Whether to override the backend if the name\n                has already been registered. Defaults to False.\n        "
        if (backend is not None):
            cls._register_backend(name, backend, force=force)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force)
            return backend_cls
        return _register

    def get(self, filepath):
        return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)
