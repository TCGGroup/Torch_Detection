from __future__ import division
import os.path as osp
import numpy as np
import cv2
from .misc import is_str, file_is_exist, exist_or_mkdir


##############################################
# image io
##############################################
def img_read(img_path, img_mode='rgb'):
    """
    Have done the read speed test between `PIL.Image.open + np.array`,
    `scipy.misc.imread` and `cv2.imread`, we found that opencv is the
    fastest image io. So we use opencv to handle the image.

    Args:
        img_path (str): the path of image file.
        img_mode (str): must be `rgb` or `bgr` (because opencv read image
            in `bgr` mode, so we convert it to `rgb` mode when read image.)

    Returns:
        img (np.ndarray): the ndarray of image read by opencv
    """
    assert is_str(img_path), "The image path must be string."
    if not file_is_exist(img_path):
        raise FileNotFoundError('{} is not exist'.format(img_path))
    assert img_mode in ['rgb', 'bgr']

    # the channel order of image is [H, W, C]
    # height, width, channel = img.shape
    img = cv2.imread(img_path)  # read by opencv, in `bgr` mode
    if img_mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img_write(img, file_path, auto_mkdir=True, img_mode='bgr'):
    """
    Write image to a file, and if `file_path` does not exist,
    when set `auto_mkdir`, this function will first make a 
    directory.

    Args:
        img (ndarray): Image array to be written into file. 
        file_path (str): The path to save the image array.
        auto_mkdir (bool): when the `file_path` does not exist,
            automatically make a directory.
        img_mode (str): must be `rgb` or `bgr` (because opencv read image
            in `bgr` mode, so we convert it to `bgr` mode when write image.)

    Returns:
        bool: write successful or not
    """
    if auto_mkdir:
        dir_name = osp.dirname(osp.abspath(file_path))
        exist_or_mkdir(dir_name)
    assert img_mode in ['rgb', 'bgr']
    if img_mode == 'bgr':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(file_path, img)


##############################################
# image normalization
##############################################
def img_normalize(img, img_mean, img_std, img_mode='rgb'):
    """
    Normalize the image by subtract the `img_mean` and 
    divide the `img_std` in the right image mode, the 
    mean and std should correspond to `img_mode`

    Args:
        img (ndarray): Image array to be normalized.
        img_mean (tuple[float]): mean value for each channel of image.
        img_std (tuple[float]): std value for each channel of image.
        img_mode (str): `rgb` or `bgr`, to specify the img mode.

    Returns:
        normed_img (ndarray): normalized image array.
    """
    assert img_mode in ['rgb', 'bgr'], "image mode must be 'rgb' or 'bgr'."
    return (img - img_mean) / img_std


def img_denormalize(img, img_mean, img_std, img_mode='rgb'):
    """
    De-normalize the image array by multiply `img_std` and add the
    `img_mean` in the right image mode, the mean and std should 
    correspond to `img_mode`
    
    Args:
        img (ndarray): Image array to be normalized.
        img_mean (tuple[float]): mean value for each channel of image.
        img_std (tuple[float]): std value for each channel of image.
        img_mode (str): `rgb` or `bgr`, to specify the img mode.

    Returns:
        normed_img (ndarray): de-normalized image array.
    """
    assert img_mode in ['rgb', 'bgr'], "image mode must be 'rgb' or 'bgr'."
    return img * img_std + img_mean


##############################################
# image resize
##############################################
def img_resize(img, size=None, scale_factor=None, return_scale=False, interpolation='nearest'):
    """
    Resize the img either given `size` or `scale_factor`. If given `size`, 
    we must set `return_scale` as `True`, if given `scale_factor`, we can
    choose `return_scale` as `True` or `False`.
    
    Args:
        img (ndarray): Image array to be resized.
        size (int or tuple[int]): if `size` is `int`, that means resize
            the short edge of image into `size`; if `size` is `tuple`, 
            that means resize image with a scale_factor use the minimum of 
            `short_edge/min(size)` and `long_edge/max(size)`
        scale_factor (float or int or tuple[int]): the scale factor used to 
            resized image. if the `scale_factor` is `tuple`, then the image
            will be resized by a randomly selected scale.
        return_scale (bool): return scale_factor or not
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos"

    Returns:
        ndarray or tuple: `resized_img` or (`resized_img`, `scale_factor`)
    """

    def _check_size_scale_factor():
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if size is not None and not return_scale:
            raise ValueError('should return scale_factor when use size')

    interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }

    _check_size_scale_factor()
    assert interpolation in ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'], \
        "interpolation {} is not supported now, please check the mode.".format(interpolation)

    h, w = img.shape[:2]
    if size is not None:
        if isinstance(size, int):
            scale_factor = size / min(h, w)
            new_h, new_w = int(h * scale_factor + 0.5), int(w * scale_factor + 0.5)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp_codes[interpolation])
        elif isinstance(size, tuple):
            scale_factor = min(min(size) / min(h, w), max(size) / max(h, w))
            new_h, new_w = int(h * scale_factor + 0.5), int(w * scale_factor + 0.5)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp_codes[interpolation])
        else:
            raise ValueError('size must be int or tuple[int], but got {}'.format(type(size)))
        return resized_img, scale_factor

    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            new_h, new_w = int(h * scale_factor + 0.5), int(w * scale_factor + 0.5)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp_codes[interpolation])
        elif isinstance(scale_factor, tuple):
            scale_factor = np.random.choice(scale_factor)
            new_h, new_w = int(h * scale_factor + 0.5), int(w * scale_factor + 0.5)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp_codes[interpolation])
        else:
            raise ValueError('scale_factor must be int, float or tuple[int], '
                             'but got {}'.format(type(scale_factor)))
        if not return_scale:
            return resized_img
        else:
            return resized_img, scale_factor


##############################################
# image flip
##############################################
def img_flip(img, flip_prob=0, direction="horizontal"):
    """
    Flip the img given the direction and flip probability.

    Args:
        img (ndarray): Image array to be flipped.
        flip_prob (float): the probability to flip the img.
        direction (str): the direction to flip the image, it must be
            one of ["horizontal", "vertical"]

    Returns:
        tuple: (flipped_img (ndarray), flipped_flag (bool)): the flipped
            image and the flipped flag.
    """
    assert direction in ["horizontal", "vertical"], \
        "the direction only support for `horizontal` and `vertical`, but got {}".format(direction)
    assert (0 <= flip_prob <= 1), "the probability to flip the image should be in the interval [0, 1]"

    flipped_flag = False
    if np.random.random() < flip_prob:
        flipped_flag = True
        if direction == "horizontal":
            flipped_img = np.flip(img, 1)
        else:
            flipped_img = np.flip(img, 0)
    else:
        flipped_img = img
    return flipped_img, flipped_flag


##############################################
# image pad
##############################################
def img_pad(img, expected_shape, pad_val=0):
    """
    Padding the image according to `expected_shape` by `pad_val`.

    Args:
        img (ndarray): the image to be padded.
        expected_shape (tuple): expected padding shape.
        pad_val (number or sequence): values to be filled in the
            padding area.

    Returns:
        padded_img (ndarray): the padded img.
    """
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img.shape[-1]
    if len(expected_shape) < len(img.shape):
        expected_shape = expected_shape + (img.shape[-1],)
    assert len(expected_shape) == len(img.shape)

    padded_img = np.empty(expected_shape, dtype=img.dtype)
    padded_img[...] = pad_val
    padded_img[:img.shape[0], :img.shape[1], ...] = img
    return padded_img


##############################################
# image crop
##############################################
def img_crop(img, size_crop, min_w=0, min_h=0):
    """
    Crop the image to `size_crop` given `min_w` and `min_h`.
    TODO: Make this function more elegant
    Args:
        img (ndarray): Image to be cropped. The channel order
            of `img` is `[height, width, channel]`
        size_crop (tuple): the image size after crop. and the
            order of `size_crop` is `[width, height]`
        min_w (int): the minimum index in the `width` side.
        min_h (int): the minimum index in the `height` side.

    Returns:
        cropped_img (ndarray): the cropped image.
    """
    assert isinstance(size_crop, tuple) and len(size_crop) == 2
    assert isinstance(min_w, int) and isinstance(min_h, int)

    cropped_width, cropped_height = size_crop
    max_w = min_w + cropped_width - 1
    max_h = min_h + cropped_height - 1
    img_h, img_w, _ = img.shape
    assert max_h <= img_h and max_w <= img_w

    cropped_img = img[min_h:max_h, min_w:max_w, ...]
    return cropped_img


##############################################
# image ratio
##############################################
def img_aspect_ratio(width, height):
    """
    Calculate the aspect ratio for image given width and height.

    Args:
        width (float): the width of image
        height (float): the height of image

    Returns:
        aspect_ratio (float): the aspect_ratio of image.
    """
    assert isinstance(width, float) and isinstance(height, float)
    return width / height


def img_aspect_ratio_flag(width, height):
    """
    Calculate the aspect ratio for image given width and height.
    then give the `flag` for the aspect ratio, when `ratio > 1`,
    we set `flag` as 1, else we set `flag` as 0.

    Args:
        width (float): the width of image
        height (float): the height of image

    Returns:
        tupel: (aspect_ratio (float), flag (int)): the aspect_ratio of
            image and the flag.
    """
    aspect_ratio = img_aspect_ratio(width, height)
    flag = int(aspect_ratio > 1)
    return aspect_ratio, flag
