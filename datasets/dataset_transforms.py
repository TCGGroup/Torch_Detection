import numpy as np
from .utils import img_read, img_normalize, img_resize, img_flip, \
    img_pad_size_divisor, \
    bbox_resize, bbox_flip, bbox_pad, mask_resize, mask_flip, mask_pad


class ImageTransforms(object):
    """
    This class is an image processing pipeline, the steps are list as follow:
        1. read an image from the path of image
        2. normalize an image given means and stds for each channel
        3. resize the image, and keep the scale factor used in resizing,
        and also
           save the shape of resized image, which will be used in bbox flip
        4. flip the image or not according to the given flip ratio
        5. pad the image to size that can divide by size divisor, keep
           the padded shape of image
        6. transpose the image channel order from (h, w, c) to (c, h, w)
    """

    def __init__(self, img_means=(0., 0., 0.), img_stds=(1., 1., 1.),
                 size_divisor=None):
        self.img_means = img_means
        self.img_stds = img_stds
        self.size_divisor = size_divisor

    def __call__(self, img_path, expected_size, flip_ratio=0):
        img = img_read(img_path)
        img = img_normalize(img, self.img_means, self.img_stds)
        img, scale_factor = img_resize(img, size=expected_size,
                                       return_scale=True)
        img_shape = img.shape
        img, flipped_flag, flipped_direction = img_flip(img, flip_ratio)
        if self.size_divisor is not None:
            img = img_pad_size_divisor(img, size_divisor=self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, \
               scale_factor, flipped_flag, flipped_direction


class BboxTransforms(object):
    """
    This class is a bbox processing pipeline, the steps are list as follow:
        1. resize the bbox given the scale factor that used in resizing image
        2. flip the bbox according to the flipped_flag and img_shape after
        image transforms
        3. pad the bbox in the first dimension if given max_num_gts of the
        whole dataset
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bbox, img_shape, scale_factor, flipped_flag,
                 flipped_direction):
        bbox = bbox_resize(bbox, scale_factor)
        bbox = bbox_flip(bbox, img_shape, flipped_flag=flipped_flag,
                         direction=flipped_direction)
        if self.max_num_gts is not None:
            bbox = bbox_pad(bbox, self.max_num_gts)
        return bbox


class MaskTransforms(object):
    """
    This class is a mask processing pipeline, the steps are list as follow:
        1. resize the mask given the scale factor that used in resizing image
        2. flip the mask according to the flipped_flag
        3. pad the mask to size that can divide by size divisor
    """

    def __call__(self,
                 masks,
                 scale_factor,
                 pad_shape,
                 flipped_flag,
                 flipped_direction):
        masks = [mask_resize(mask, scale_factor=scale_factor)
                 for mask in masks]
        masks = [mask_flip(mask, flipped_flag=flipped_flag,
                           direction=flipped_direction) for mask in masks]
        padded_masks = [
            mask_pad(mask, expected_shape=pad_shape[:2]) for mask in masks]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks
