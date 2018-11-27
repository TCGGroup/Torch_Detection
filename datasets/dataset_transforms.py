import numpy as np
from .utils import img_read, img_normalize, img_resize, img_flip, \
    img_pad_size_divisor, bbox_resize, bbox_flip, bbox_pad, \
    mask_resize, mask_flip, mask_pad


class ImageTransforms(object):
    """
    This class is an image processing pipeline, the steps are list as follow:
        1. read an image from the path of image
        2. normalize an image given means and stds for each channel
        3. resize the image, and keep the scale factor used in resizing, and
           also save the shape of resized image, which will be used in bbox
           flip
        4. flip the image or not according to the given flip ratio
        5. pad the image to size that can divide by size divisor, keep
           the padded shape of image
        6. transpose the image channel order from (h, w, c) to (c, h, w)
    """

    def __init__(self,
                 img_means=(0., 0., 0.),
                 img_stds=(1., 1., 1.),
                 size_divisor=None):
        self.img_means = img_means
        self.img_stds = img_stds
        self.size_divisor = size_divisor

    def __call__(self, img_path, expected_size, flip_ratio=0):
        # TODO: duplicated read if we use multi-times transform for an image
        img = img_read(img_path)
        img = img_normalize(img, self.img_means, self.img_stds)
        img, scale_factor = img_resize(
            img, size=expected_size, return_scale=True)
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
    """

    def __call__(self,
                 bbox,
                 img_shape,
                 scale_factor,
                 flipped_flag,
                 flipped_direction):
        bbox = bbox_resize(bbox, scale_factor)
        bbox = bbox_flip(bbox,
                         img_shape,
                         flipped_flag=flipped_flag,
                         direction=flipped_direction)
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
        masks = [
            mask_resize(mask, scale_factor=scale_factor)
            for mask in masks
        ]
        masks = [
            mask_flip(mask,
                      flipped_flag=flipped_flag,
                      direction=flipped_direction)
            for mask in masks
        ]
        padded_masks = [
            mask_pad(mask, expected_shape=pad_shape[:2])
            for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class BackgroundErasing(object):
    """
    This class is to erasing background in the image with the help of ground
    truth bounding boxes. The processing pipeline is as follow:
        1. expand the ground truth boxes with expansion `cell_size / 2`
        2. cut the image into grids, and each cell is a square except the cell
           in the right edge
        3. calculate the overlaps between each cell and each ground truth box,
           for a cell, if there exist an overlap with any ground truth, we
           mark this cell as `foreground cell`, and those do not has an
           overlap with any ground truth box, we mark as `background cell`
        4. then we choose a number of `number_background_cell * random_ratio`
           among these `background cell`
        5. we set the image into total black (that means inplace with 0) in the
           exact position of the chosen cells
    TODO: make this class more elegant
    """

    def __call__(self, img, img_shape, bbox, cell_size=32, random_ratio=0.5):
        num_y_cell = np.ceil(img_shape[0] / cell_size)
        num_x_cell = np.ceil(img_shape[1] / cell_size)
        shift_ctrx = np.arange(0, num_x_cell) * cell_size
        shift_ctry = np.arange(0, num_y_cell) * cell_size
        shift_ctrx, shift_ctry = np.meshgrid(shift_ctrx, shift_ctry)
        shifts = np.vstack(
            (shift_ctrx.ravel(), shift_ctry.ravel())).transpose()
        cells = np.hstack((shifts, (shifts + cell_size - 1)))
        cells[..., 0::2] = np.clip(
            cells[..., 0::2], 0, img_shape[1] - 1)
        cells[..., 1::2] = np.clip(
            cells[..., 1::2], 0, img_shape[0] - 1)

        expand_bbox = bbox.copy()
        expand_bbox[..., :2] = bbox[..., :2] - cell_size // 2 + 1
        expand_bbox[..., 2:] = bbox[..., 2:] + cell_size // 2 - 1
        expand_bbox[..., 0::2] = np.clip(
            expand_bbox[..., 0::2], 0, img_shape[1] - 1)
        expand_bbox[..., 1::2] = np.clip(
            expand_bbox[..., 1::2], 0, img_shape[0] - 1)

        valid_flag = self._has_overlap(cells, expand_bbox)
        valid_cells = cells[valid_flag]
        if len(valid_cells) > 0:
            inds = np.arange(len(valid_cells))
            inds_choice = np.random.choice(
                inds,
                size=int(np.ceil(len(valid_cells) * random_ratio)),
                replace=False)
            choiced_cells = valid_cells[inds_choice]
            img = self.fill_black(img, choiced_cells)
        return img

    @staticmethod
    def _has_overlap(bbox1, bbox2):
        lt = np.maximum(bbox1[:, None, :2], bbox2[:, :2])
        rb = np.minimum(bbox1[:, None, 2:], bbox2[:, 2:])
        wh = ((rb - lt) > 0).astype(np.uint8)
        has_overlap = ((wh[..., 0] * wh[..., 1]) > 0).astype(np.uint8)
        valid_flag = has_overlap.sum(axis=1) == 0
        return valid_flag

    @staticmethod
    def fill_black(img, fill_cells):
        for cell in fill_cells:
            img[:, int(cell[1]):int(cell[3] + 1),
                int(cell[0]):int(cell[2] + 1)] = 0
        return img
