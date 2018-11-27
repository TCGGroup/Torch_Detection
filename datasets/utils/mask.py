import numpy as np
import cv2
from .image import img_write, img_visualize, img_resize, img_crop, img_pad


##############################################
# mask io
##############################################
def mask_parse(annotation, gt_masks, gt_mask_polys, gt_poly_lens, dataset):
    """
    Parse ground-truth mask in an annotation dict. There is no return in this
    function, because the `gt_masks`, `gt_mask_polys`, `gt_poly_lens` are
    lists, and if they have append element in this function, the change will
    be with them automatically, do not need to return.
    And because the mask is the whole image size, and actually, when we get the
    mask target of the object, we do not need the whole mask, we just need the
    exact piece of mask, so we do not seed the mask to GPU, we re-generate the
    mask target in the `mask_head` module, and seed it to the GPU where the
    `predicted_masks` is.

    Args:
        annotation (dict): The annotation dict for an image.
        gt_masks (list): The list of ground truth masks, each element in this
            list is will generate a ground truth binary mask
        gt_mask_polys (list): The list of ground truth poly coordinates, each
            element in this list is a couple of coordinates for each point in
            the polygons.
        gt_poly_lens (list): Each element in this list is number of points in
            the polygons.
        dataset (:obj:dataset): The dataset object, e.g. COCO dataset object.
    """

    # annToMask: first use `annToRLE` to compress polygons, and then use
    # `decode` convert the compressed polygon into a binary mask.
    gt_masks.append(dataset.annToMask(annotation))
    mask_polys = [
        p for p in annotation['segmentation'] if len(p) >= 6
    ]  # valid polygons have >= 3 points (6 coordinates)
    poly_lens = [len(p) for p in mask_polys]
    gt_mask_polys.append(mask_polys)
    gt_poly_lens.extend(poly_lens)


##############################################
# mask visualize
##############################################
def mask_visualize(img_array, masks, inds,
                   mask_color=(0, 255, 0), alpha=0.5,
                   show=True, win_name='', wait_time=0, out_file=None):
    """
    Visualize the predicted mask with `inds`

    Args:
        img_array (ndarray): The image with `bbox` and `text` to be displayed.
        masks (ndarray, np.uint8): The predicted masks for each `bbox`
        inds (ndarray of bool or `[]`): The `inds` get from
            `:func:bbox_visualize`
        mask_color (tuple): color of bbox lines.
        alpha (float): the opacity of the mask contour
        show (bool):Whether to show the image
        win_name (str): The window name
        wait_time (int): value of waiting time to display.
        out_file (str or None): The filename to write the image.
    """
    assert masks.ndim == 3
    assert inds.ndim == 1
    masks = masks.astype(np.uint8)

    output_img = img_array.copy()
    if len(inds) > 0:
        masks = masks[inds, ...]

    for mask in masks:
        _, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(img_array, pts=contours, color=mask_color)
    # `:func:cv2.addWeighted`: dst = src1 * alpha + src2 * beta + gamma
    cv2.addWeighted(src1=img_array, alpha=alpha, src2=output_img,
                    beta=1 - alpha, gamma=0, dst=output_img)

    if show:
        img_visualize(output_img, win_name, wait_time)
    if out_file is not None:
        img_write(output_img, out_file)


##############################################
# mask resize
##############################################
def mask_resize(mask,
                scale_factor=None,
                return_scale=False,
                interpolation='nearest'):
    """
    Resize the binary mask according to the `scale_factor` that used to
    resize the image. Because `mask` is a binary image, so this function is
    the Same as `:func:img_resize`

    Args:
        mask (ndarray, np.uint8): Binary image for object, the shape of
        `mask` is `[h, w]`
        scale_factor: same as `:func:img_resize`
        return_scale: same as `:func:img_resize`
        interpolation: same as `:func:img_resize`

    Returns:
        resized_mask (ndarray, np.uint8): resized binary mask
    """
    assert mask.ndim == 2
    mask = mask.astype(np.uint8)

    resized_mask = img_resize(mask,
                              scale_factor=scale_factor,
                              return_scale=return_scale,
                              interpolation=interpolation)
    return resized_mask


##############################################
# mask flip
##############################################
def mask_flip(mask, flipped_flag=True, direction="horizontal"):
    """
    Flip the binary mask when the `flipped_flag` is True. And flip the
    binary mask in the `direction` of flipping image.

    Args:
        mask (ndarray, np.uint8): Binary image for object, the shape of
        `mask` is `[h, w]`
        flipped_flag (bool): Whether the image has flipped or not, this flag
            comes from the returns of `:func:img_flip`
        direction (str): same as the param in `:func:img_flip`

    Returns:
        flipped_mask (ndarray, np.uint8): flipped binary mask
    """
    assert mask.ndim == 2
    mask = mask.astype(np.uint8)

    if not flipped_flag:
        return mask
    else:
        if direction == 'horizontal':
            flipped_mask = np.flip(mask, 1)
        else:
            flipped_mask = np.flip(mask, 0)
    return flipped_mask


##############################################
# mask crop
##############################################
def mask_crop(mask, size_crop, min_w=0, min_h=0):
    """
    Crop the binary mask, same as `:func:img_crop`

    Args:
        mask (ndarray, np.uint8): Binary image for object, the shape of
        `mask` is `[h, w]`
        size_crop (tuple): same as `:func:img_crop`
        min_w (int): same as `:func:img_crop`
        min_h (int): same as `:func:img_crop`

    Returns:
        cropped_mask (ndarray, np.uint8): cropped binary mask
    """
    assert mask.ndim == 2
    mask = mask.astype(np.uint8)

    return img_crop(mask, size_crop, min_w=min_w, min_h=min_h)


##############################################
# mask pad
##############################################
def mask_pad(mask, expected_shape, pad_val=0):
    """
    Pad the binary mask, same as `:func:img_pad`

    Args:
        mask (ndarray, np.uint8): Binary image for object, the shape of
        `mask` is `[h, w]`
        expected_shape (tuple): same as `:func:img_pad`
        pad_val (number or sequence): same as `:func:img_pad`

    Returns:
        padded_mask (ndarray, np.uint8): padded binary mask
    """
    assert mask.ndim == 2
    mask = mask.astype(np.uint8)

    return img_pad(mask, expected_shape, pad_val=pad_val)
