from __future__ import division
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
        img (ndarray): the ndarray of image read by opencv
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
        exist_or_mkdir(file_path)
    assert img_mode in ['rgb', 'bgr']
    if img_mode == 'bgr':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(file_path, img)


##############################################
# image visualize
##############################################
def img_visualize(img_array, win_name='', wait_time=0, single_img=True):
    """
    Show an image given its image array.

    Args:
        img_array (ndarray): The image array to be displayed.
        win_name (str): The window name.
        wait_time (int): value of waiting time to display.
        single_img (bool): Whether to show a single image or a video,
            if `single_img`, destroy the window after `quit`, else,
            destroy all windows after the video is release.
    """
    assert is_str(win_name), "window name must be string"
    cv2.imshow(win_name, img_array)
    cv2.waitKey(wait_time)  # use `q` to exit the showing window.
    if single_img:
        cv2.destroyWindow(win_name)


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
def img_resize(img, size=None, scale_factor=None, return_scale=False,
               interpolation='nearest'):
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
            raise ValueError(
                'only one of size or scale_factor should be defined')
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
    assert interpolation in \
        ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'], \
        "interpolation {} is not supported now, " \
        "please check the mode.".format(interpolation)

    h, w = img.shape[:2]
    if size is not None:
        if isinstance(size, int):
            scale_factor = size / min(h, w)
            new_h, new_w = int(h * scale_factor + 0.5), \
                int(w * scale_factor + 0.5)
            resized_img = cv2.resize(
                img, (new_w, new_h), interpolation=interp_codes[interpolation])
        elif isinstance(size, tuple):
            scale_factor = min(min(size) / min(h, w), max(size) / max(h, w))
            new_h, new_w = int(h * scale_factor + 0.5), \
                int(w * scale_factor + 0.5)
            resized_img = cv2.resize(
                img, (new_w, new_h), interpolation=interp_codes[interpolation])
        else:
            raise ValueError(
                'size must be int or tuple[int], '
                'but got {}'.format(type(size)))
        return resized_img, scale_factor

    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            new_h, new_w = int(h * scale_factor + 0.5), \
                int(w * scale_factor + 0.5)
            resized_img = cv2.resize(
                img, (new_w, new_h), interpolation=interp_codes[interpolation])
        elif isinstance(scale_factor, tuple):
            scale_factor = np.random.choice(scale_factor)
            new_h, new_w = int(h * scale_factor + 0.5), \
                int(w * scale_factor + 0.5)
            resized_img = cv2.resize(
                img, (new_w, new_h), interpolation=interp_codes[interpolation])
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
        tuple: (flipped_img (ndarray), flipped_flag (bool), direction (str)):
            the flipped image, the flipped flag and the flipped direction.
    """
    assert direction in ["horizontal", "vertical"], \
        "the direction only support for `horizontal` and `vertical`, " \
        "but got {}".format(direction)
    assert (0 <= flip_prob <= 1), \
        "the probability to flip the image should be in the interval [0, 1]"

    flipped_flag = False
    if np.random.random() < flip_prob:
        flipped_flag = True
        if direction == "horizontal":
            flipped_img = np.flip(img, 1)
        else:
            flipped_img = np.flip(img, 0)
    else:
        flipped_img = img
    return flipped_img, flipped_flag, direction


##############################################
# image rotate
##############################################
def img_rotate(img, angle, center=None, scale=1.0, border_value=0,
               auto_bound=False):
    """
    Rotate an image according to given parameters.

    Args:
        img (ndarray): Image array to be rotated.
        angle (float): Rotation degree, positive value mean clockwise rotation.
        center (tuple): Center of the rotation in the image, by default it is
            the center of the image.
        scale (float): Isotropic scale factor. normally, we set `scale=1.0`
        border_value (int or tuple): Border value. If `int`, means use `red` to
            fill the border, and if `tuple`, must be `rgb mode`, `(r, g, b)`.
        auto_bound (bool): Whether to adjust the image size tp cover the whole
            rotated image.

    Returns:
        rotated_img (ndarray): the rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError("`auto_bound` conflicts with `center`, "
                         "we only automatically adjust the image size "
                         "when the center is the image center.")
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)
    # in this function, positive angle means counter-clockwise rotation
    # doc: https://docs.opencv.org/2.4/modules/imgproc/doc
    # /geometric_transformations.html#cv2.getRotationMatrix2D
    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated_img = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated_img


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


def img_pad_size_divisor(img, size_divisor, pad_val=0):
    """
    Padding the image to the size that can be divide by `size_divisor`.

    Args:
        img (ndarray): Image array to be padded.
        size_divisor (int): the number that the padded image can be
            divided by `size_divisor`
        pad_val (number or sequence): The same as :func:img_pad

    Returns:
        padded_img (ndarray): the padded image array
    """
    assert isinstance(size_divisor, int)

    img_h, img_w, _ = img.shape
    padded_img_h = int(np.ceil(img_h / size_divisor) * size_divisor)
    padded_img_w = int(np.ceil(img_w / size_divisor) * size_divisor)
    padded_shape = (padded_img_h, padded_img_w)

    padded_img = img_pad(img, padded_shape, pad_val=pad_val)
    return padded_img


##############################################
# image crop
##############################################
def img_crop(img, size_crop, min_w=0, min_h=0):
    """
    Crop the image to `size_crop` given `min_w` and `min_h`.

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
    assert min_w >= 0 & min_h >= 0

    cropped_width, cropped_height = size_crop
    max_w = min_w + cropped_width - 1
    max_h = min_h + cropped_height - 1
    img_h, img_w, _ = img.shape
    assert max_h <= img_h and max_w <= img_w

    cropped_img = img[min_h:(max_h + 1), min_w:(max_w + 1), ...]
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
