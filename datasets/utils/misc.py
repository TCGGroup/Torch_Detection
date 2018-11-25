import six
import os
import os.path as osp


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
