import numpy as np
from pycocotools.coco import COCO
from .base_dataset import BaseDataset
from .utils import bbox_parse, mask_parse


class CocoDataset(BaseDataset):
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        # load categories
        cat_ids = self.coco.getCatIds()
        self.classes = self.coco.loadCats(cat_ids)
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(cat_ids)
        }
        # load img_ids
        self.img_ids = self.coco.getImgIds()
        # sort the image ids, so we can get the same image order every time
        self.img_ids.sort()
        img_infos = []
        for img_id in self.img_ids:
            info = self.coco.loadImgs([img_id])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths"""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        ann_ids = self.coco.getAnnIds(imgIds=[self.img_infos[idx]['id']])
        anns = self.coco.loadAnns(ann_ids)

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided for mask in coco.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float
        if self.with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        for i, ann in enumerate(anns):
            bbox_parse_return = bbox_parse(ann, gt_bboxes, gt_labels,
                                           gt_bboxes_ignore, self.cat2label)
            if bbox_parse_return is False:
                continue
            if self.with_mask:
                mask_parse(ann, gt_masks, gt_mask_polys,
                           gt_poly_lens, self.coco)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore
        )

        if self.with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann
