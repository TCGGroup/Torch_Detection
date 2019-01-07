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


def exist_or_mkdir(file_path, dir=False):
    """
    check the directory name is already exist or not,
    if exist, just pass it, else make a new directory
    named `dir_name`.
    """
    if not dir:
        dir_name = osp.dirname(osp.abspath(file_path))
    else:
        dir_name = osp.abspath(file_path)
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


def random_scale(img_expected_sizes, mode='range'):
    """
    Random select an expected_size from `img_expected_sizes` given the chosen
    mode. If the mode is `value`, we just choose one element in the list, and
    if the mode is `range`, we first get the range of `long edge` and `short
    edge`, then choose one `long edge` and `short edge` respectively, at last,
    we return tuple of `(long_chosen, short_chosen)`.

    Args:
        img_expected_sizes (list[tuple]): the list of image sizes in the
            format of `[(long1, short1), (long2, short2), ...]`
        mode (str): the mode to choose the expected_size

    Returns:
        expected_size (tuple)
    """
    assert is_list_of(img_expected_sizes, tuple)
    assert mode in ['range', 'value'], \
        "we only support `['range', 'value']` modes, but got {}".format(mode)

    if len(img_expected_sizes) == 1:
        expected_size = img_expected_sizes[0]
    elif len(img_expected_sizes) == 2:
        if mode == 'value':
            ind = np.random.randint(0, len(img_expected_sizes))
            expected_size = img_expected_sizes[ind]
        else:
            long_tuple, short_tuple = zip(*img_expected_sizes)
            min_long, max_long = min(long_tuple), max(long_tuple)
            min_short, max_short = min(short_tuple), max(short_tuple)
            long_chosen = np.random.randint(min_long, max_long + 1)
            short_chosen = np.random.randint(min_short, max_short + 1)
            expected_size = (long_chosen, short_chosen)
    else:
        mode = 'value'
        assert mode == 'value', \
            "only `value` mode is supported " \
            "in the case of more than two image sizes"
        ind = np.random.randint(0, len(img_expected_sizes))
        expected_size = img_expected_sizes[ind]
    return expected_size
