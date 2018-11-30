import sys
from datasets.utils import is_str


def obj_from_dict(args_dict, parent=None, additional_dict=None):
    """
    Initialize an parent object given dict arguments and additional dict.

    This dict must contain the key `type`, which indicates the object type, it
    can be either a string or type, such as "list" and `list`. Remaining fields
    are treated as the arguments for constructing the object.

    Args:
        args_dict (dict): object types and arguments.
        parent (`class:module`): Module which may containing expected object
            classes.
        additional_dict (dict, optional): additional fields for initializing
            the object.

    Returns:
        :object: object build from the arguments.
    """
    assert isinstance(args_dict, dict) and 'type' in args_dict
    assert isinstance(additional_dict, dict) or additional_dict is None
    args = args_dict.copy()
    obj_type = args.pop('type')
    if is_str(obj_type):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if additional_dict is not None:
        for name, value in additional_dict:
            args.setdefault(name, value)
    return obj_type(**args)
