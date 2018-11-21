import numpy as np
from .image import img_crop


##############################################
# bbox io
##############################################
def bbox_parse(annotation, gt_bboxes, gt_labels, gt_bboxes_ignore, cat2label):
    """
    Parse ground-truth box in an annotation dict. There is no return in this
    function, because the `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore` are
    lists, and if they have append element in this function, the change will
    be with them automatically, do not need to return.

    Args:
        annotation (dict): The annotation dict for an image.
        gt_bboxes (list): The list of ground truth boxes, each element
            in this list is a ground truth box `[x1, y1, x2, y2]`
        gt_labels (list): The list of ground truth boxes' labels, each
            element in this list is a numerical label for the specific
            gt-box.
        gt_bboxes_ignore (list): The list of crowded ground truth boxes,
            and we will ignore the crowded ground truth boxes. Each element
            in this list is a ignored ground truth box `[x1, y1, x2, y2]`
        cat2label (dict): The dict that save the matching between the category
            id in the dataset and the numerical label.
    """
    assert len(gt_bboxes) == len(gt_labels), \
        "The length of gt_bboxes and gt_labels must match."
    # if this annotation is marked as `ignore`, or `area <= 0`
    # we return `-1` and  pass this image. The loop continues
    if annotation.get('ignore', False):
        return -1
    x1, y1, w, h = annotation['bbox']
    if annotation['area'] <= 0 or w < 1 or h < 1:
        return -1

    # change `bbox` mode from `xywh` to `xyxy`
    bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
    if annotation['iscrowd']:
        gt_bboxes_ignore.append(bbox)
    else:
        gt_bboxes.append(bbox)
        gt_labels.append(cat2label[annotation['category_id']])


##############################################
# bbox normalize
##############################################
def bbox_normalize(bbox, means=[0, 0, 0, 0], stds=[1., 1., 1., 1.]):
    """
    TODO: there are `numpy` and `Tensor`, transform all to `Tensor`
    Normalize the bbox by using means and stds.

    Args:
        bbox (Tensor): bbox tensor to be normalized, which has shape of `A x 4`
        means (list): list of means for each element in a bbox, the length is 4.
        stds (list): list of std for each element in a bbox, the length is 4

    Returns:
        normalized_bbox (Tensor): normalized bbox.
    """
    assert bbox.shape[1] == len(means) == len(stds) == 4
    # convert means and stds into tensor like bbox, and expand the first channel
    means = bbox.new_tensor(means).unsqueeze(0)
    stds = bbox.new_tensor(stds).unsqueeze(0)
    # we do not need to backward in this function
    # so we use in-place operation here to save memory
    # Attention: because we use `in-place operation`, so actually
    # the origin value of `bbox` will also change.
    return bbox.sub_(means).div_(stds)


def bbox_denormalize(bbox, means=[0, 0, 0, 0], stds=[1., 1., 1., 1.]):
    """
    De-normalize the bbox by using means and stds.

    Args:
        bbox (Tensor): bbox tensor to be de-normalized, which has shape of `A x 4`
            (regression class agnostic) or has shape of `A x 4C` (regression class
            specific)
        means (list): list of means for each element in a bbox, the length is 4.
        stds (list): list of std for each element in a bbox, the length is 4

    Returns:
        de-normalized_bbox (Tensor): De-normalized bbox.
    """
    assert bbox.shape[1] % 4 == 0
    assert len(means) == len(stds) == 4
    # convert means and stds into tensor like bbox
    # and expand the first channel, repeat the second channel
    means = bbox.new_tensor(means).repeat(1, bbox.shape[1] // 4)
    stds = bbox.new_tensor(stds).repeat(1, bbox.shape[1] // 4)
    # here we do not use in-place operation
    # because we have to backward loss in this function
    denormalized_bbox = bbox * stds + means
    return denormalized_bbox


##############################################
# bbox resize
##############################################
def bbox_resize(bbox, scale_factor):
    """
    Resize the bbox according to the `scale_factor` saved for
    resizing the image.

    Args:
        bbox (ndarray): All gt boxes in an image, and the shape of `bbox`
            is `K x 4`
        scale_factor (float or int): The `scale_factor` used in resizing
            the image.

    Returns:
        resized_bbox (ndarray): The resized bbox by `scale_factor`
    """
    assert isinstance(scale_factor, (int, float))
    resized_bbox = bbox * scale_factor
    return resized_bbox


##############################################
# bbox flip
##############################################
def bbox_flip(bbox, img_shape, flipped_flag=True, direction="horizontal"):
    """
    Flip the `bbox` given the direction and flip flag. And
    pay attention, the direction and flip flag must correspond
    with the `img_flip`.
    TODO: add mode attribute for `bbox`, supported mode: `xyxy`

    Args:
        bbox (ndarray): All gt boxes in an image, and the shape of `bbox`
            is `K x 4`, mode of bbox is 'xyxy'
        img_shape (tuple): the tuple of `(height, width)` of image.
        flipped_flag (bool): Whether the image has flipped or not, this flag
            comes from the returns of `:func:img_flip`
        direction (str): same as the param in `:func:img_flip`

    Returns:
        flipped_bbox (ndarray): flipped bbox.
    """
    assert bbox.shape[-1] == 4
    assert isinstance(img_shape, tuple) and len(img_shape) == 2
    assert direction in ["horizontal", "vertical"]

    if not flipped_flag:
        return bbox
    else:
        if direction == 'horizontal':
            w = img_shape[1]
            flipped_bbox = bbox.copy()
            flipped_bbox[..., 0] = w - bbox[..., 2] - 1
            flipped_bbox[..., 2] = w - bbox[..., 0] - 1
        else:
            h = img_shape[0]
            flipped_bbox = bbox.copy()
            flipped_bbox[..., 1] = h - bbox[..., 3] - 1
            flipped_bbox[..., 3] = h - bbox[..., 1] - 1
        return flipped_bbox


##############################################
# bbox crop
##############################################
def bbox_crop(img, bbox, label, size_crop):
    """
    Crop the image according to `bbox`, we choose part of image
    with the size of `size_crop` that contains most of `bbox` in
    the part.
    crop an `img` according to `bbox`:
    1. get the `bbox_width' and `bbox_height` in the `bbox`
    2. compare `bbox_width` with `size_crop[0]`:
            - if `size_crop[0] > bbox_width`, random choose `min_w`
            - else just choose the `min_bbox_w` as `min_w`
    3. compare `bbox_height` with `size_crop[1]`:
            - if `size_crop[1] > bbox_height`, random choose 'min_h'
            - else just choose the `min_bbox_h` as `min_h`
    4. filter the `bbox` that `area = 0`, and also the `label`
    TODO: better method to choose crop regions in the image
    TODO: add mode attribute for `bbox`, supported mode: `xyxy`

    Args:
        img (ndarray): Image to be cropped. The channel order
            of `img` is `[height, width, channel]`
        bbox (ndarray): All gt boxes in an image, and the shape
            of `bbox` is `K x 4`, mode of bbox is `xyxy`
        label (ndarray): The label of all gt boxes, and the shape
            of `label` is `K`
        size_crop (tuple): the image size after crop. and the
            order of `size_crop` is `[width, height]`

    Returns:
        tuple: (cropped_img (ndarray), cropped_bbox (ndarray))
    """
    assert bbox.shape[-1] == 4

    min_bbox_w, max_bbox_w = np.min(bbox[..., 0]), np.max(bbox[..., 2])
    min_bbox_h, max_bbox_h = np.min(bbox[..., 1]), np.max(bbox[..., 3])
    bbox_width = max_bbox_w - min_bbox_w + 1
    bbox_height = max_bbox_h - min_bbox_h + 1

    img_h, img_w, _ = img.shape
    cropped_width, cropped_height = size_crop
    if cropped_width < bbox_width:
        min_w = int(min_bbox_w)
    else:
        min_crop_w = max(max_bbox_w - cropped_width + 1, 0)
        max_crop_w = min(img_w - cropped_width, min_bbox_w)
        min_w = np.random.choice(range(int(min_crop_w), int(max_crop_w) + 1))
        # type `np.int32` is not as same as `int`
        min_w = int(min_w)
    if cropped_height < bbox_height:
        min_h = int(min_bbox_h)
    else:
        min_crop_h = max(max_bbox_h - cropped_height + 1, 0)
        max_crop_h = min(img_h - cropped_height, min_bbox_h)
        min_h = np.random.choice(range(int(min_crop_h), int(max_crop_h) + 1))
        # type `np.int32` is not as same as `int`
        min_h = int(min_h)

    cropped_img = img_crop(img, size_crop, min_w=min_w, min_h=min_h)
    cropped_bbox = bbox.copy()
    cropped_bbox[..., 0::2] = np.clip(cropped_bbox[..., 0::2] - min_w, 0, cropped_width - 1)
    cropped_bbox[..., 1::2] = np.clip(cropped_bbox[..., 1::2] - min_h, 0, cropped_height - 1)

    # filter bbox and label
    invalid = (cropped_bbox[..., 0] == cropped_bbox[..., 2]) | (cropped_bbox[..., 1] == cropped_bbox[..., 3])
    valid_inds = np.nonzero(~invalid)[0]
    if len(valid_inds) < len(cropped_bbox):
        cropped_bbox = cropped_bbox[valid_inds]
        label = label[valid_inds]
    return cropped_img, cropped_bbox, label


##############################################
# bbox change mode
##############################################
def bbox_convert_mode(bbox, mode='xywh2xyxy'):
    """
    Change the mode of `bbox`, because we use different `bbox` mode in
    different situation, e.g., we use `xywh` in annotations, we use `xyxy`
    in `bbox transforms`.
    Currently, we only consider `bbox` mode change in dataset processing stage.
    TODO: consider `:Tensor:bbox` mode `cxcywh` in anchor, proposals, rois and deltas.
    TODO: reference `https://github.com/facebookresearch/maskrcnn-benchmark` and use `class` to build all the functions

    Args:
        bbox (ndarray): All gt boxes in an image, and the shape
            of `bbox` is `K x 4`
        mode (str): the mode change of `bbox`, must in `['xywh2xyxy', 'xyxy2xywh']`

    Returns:
        mode_bbox (ndarray): the `bbox` in the another mode
    """
    assert mode in ['xywh2xyxy', 'xyxy2xywh']
    a = bbox[..., :2]
    b = bbox[..., 2:]
    if mode == 'xyxy2xywh':
        mode_bbox = np.hstack([a, b - a + 1])
    else:
        mode_bbox = np.hstack([a, a + b - 1])
    return mode_bbox
