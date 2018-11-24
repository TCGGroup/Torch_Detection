from abc import ABCMeta, abstractmethod
import pickle
import json
from .misc import is_str, file_is_exist, exist_or_mkdir


class BasicFileHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self, filepath, mode='r', **kwargs):
        pass

    @abstractmethod
    def dump(self, obj, filepath, mode='w', **kwargs):
        pass


class PickleHandler(BasicFileHandler):
    def load(self, filepath, mode='rb', **kwargs):
        with open(filepath, mode=mode) as f:
            return pickle.load(f, **kwargs)

    def dump(self, obj, filepath, mode='wb', **kwargs):
        with open(filepath, mode=mode) as f:
            kwargs.setdefault('protocol', 2)
            pickle.dump(obj, f, **kwargs)


class JsonHandler(BasicFileHandler):
    def load(self, filepath, mode='r', **kwargs):
        with open(filepath, mode=mode) as f:
            return json.load(f, **kwargs)

    def dump(self, obj, filepath, mode='w', **kwargs):
        with open(filepath, mode=mode) as f:
            json.dump(obj, f, **kwargs)


file_handlers = {
    'pkl': PickleHandler(),
    'json': JsonHandler()
}


def load(filepath, file_format=None, **kwargs):
    """
    Load from `pkl/json` files.

    Args:
        filepath (str): file path to be loaded.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "pickle/pkl".
        **kwargs: parameters for `:func:pickle.load` or `:func:json.load`

    Returns:
        The content from the file.
    """
    assert is_str(filepath), "The filepath must be string, but got type {}".format(type(filepath))
    if not file_is_exist(filepath):
        raise FileExistsError('{}, file is not exist'.format(filepath))

    if file_format is None:
        file_format = filepath.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    obj = handler.load(filepath, **kwargs)
    return obj


def dump(obj, filepath, file_format=None, **kwargs):
    """
    Dump data to `pkl/json` files.

    Args:
        obj (any): The python obj to be dumped.
        filepath (str): The output file path to save the data.
        file_format (str, optional): same as `:func:load`
        **kwargs: parameters for `:func:pickle.dump` or `:func:json.dump`
    """
    assert is_str(filepath), "The filepath must be string, but got type {}".format(type(filepath))
    exist_or_mkdir(filepath)

    if file_format is None:
        file_format = filepath.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    handler.dump(obj, filepath, **kwargs)
