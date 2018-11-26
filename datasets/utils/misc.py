import six
import os
import os.path as osp
import numpy as np
import torch
from collections import Sequence


def is_str(x):
    """Whether the input is a string instance."""
    return isinstance(x, six.string_types)


def file_is_exist(file_path):
    """Whether the input file exist."""
    return osp.isfile(file_path)


def exist_or_mkdir(file_path):
    """
    check the directory name is already exist or not,
    if exist, just pass it, else make a new directory
    named `dir_name`.
    """
    dir_name = osp.dirname(osp.abspath(file_path))
    dir_name = osp.expanduser(dir_name)
    if not osp.isdir(dir_name):
        # the difference between `os.mkdir` and `os.makedirs`
        # is, `os.mkdir` only make last level directory, but
        # `os.makedirs` make multi-level directories recurrently
        os.makedirs(dir_name)


def is_list_of(seq, check_type):
    """
    check if the `obj` is a list of `check_type` data.
    """
    if not isinstance(seq, list):
        return False
    else:
        for item in seq:
            if not isinstance(item, check_type):
                return False
        return True

def to_tensor(data):
    """
    Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are:
        1. `numpy.ndarray`,
        2. `torch.Tensor`,
        3. `Sequence`,
        4. `int`,
        5. `float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be '
                        'converted to tensor.'.format(type(data)))

def random_scale(img_scales, mode='range'):
    pass
