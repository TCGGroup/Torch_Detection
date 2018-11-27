import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
from .utils import file_is_exist, exist_or_mkdir, dump
from .base_dataset import BaseDataset


class VOCDataset(BaseDataset):
    """
    The directory tree for `voc` dataset. And you can get the data format in
    the `xml` file at here:
    https://blog.csdn.net/weixin_35653315/article/details/71028523

    voc
    ├── voc2007
    │   ├── annotations
    │   │   ├── 009963.xml
    │   │   ├── 009962.xml
    │   │   ├── ...
    │   ├── JPEGImages
    │   │   ├── 009963.jpg
    │   │   ├── 009962.jpg
    │   │   ├── ...
    │   ├── ImageSets
    │   │   ├── Main
    │   │   │   ├── trainval.txt
    │   │   │   ├── test.txt
    │   │   ├── ...
    │   ├── ...
    │
    ├── voc2012
    │   ├── annotations
    │   │   ├── 2012_004331.xml
    │   │   ├── 2012_004330.xml
    │   │   ├── ...
    │   ├── JPEGImages
    │   │   ├── 2012_004331.jpg
    │   │   ├── 2012_004331.jpg
    │   │   ├── ...
    │   ├── ImageSets
    │   │   ├── Main
    │   │   │   ├── trainval.txt
    │   │   ├── ...
    │   ├── ...
    """

    def __init__(self,
                 cache_dir='data/cache/',
                 dataset_scope='voc07',
                 dataset_root='data/voc/voc2007/',
                 img_means=(0, 0, 0),
                 img_stds=(1., 1., 1.),
                 img_expected_sizes=(1000, 600),
                 size_divisor=None,
                 flip_ratio=0,
                 be_cell_size=32,
                 be_random_ratio=0.5,
                 proposal_file=None,
                 num_max_proposals=1000,
                 with_mask=False,
                 with_crowd=False,
                 with_label=True,
                 test_mode=False,
                 with_background_erasing=False,
                 debug=False
                 ):
        assert dataset_scope in ['voc07', 'voc12', 'voc07+12']
        ann_file, img_prefix = self._parse_voc(cache_dir=cache_dir,
                                               dataset_scope=dataset_scope,
                                               dataset_root=dataset_root,
                                               test_mode=test_mode)
        super(VOCDataset, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            img_means=img_means,
            img_stds=img_stds,
            img_expected_sizes=img_expected_sizes,
            size_divisor=size_divisor,
            flip_ratio=flip_ratio,
            be_cell_size=be_cell_size,
            be_random_ratio=be_random_ratio,
            proposal_file=proposal_file,
            num_max_proposals=num_max_proposals,
            with_mask=with_mask,
            with_crowd=with_crowd,
            with_label=with_label,
            test_mode=test_mode,
            with_background_erasing=with_background_erasing,
            debug=debug)

    def _parse_voc(self, cache_dir, dataset_scope, dataset_root, test_mode):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        class_to_cat = {
            cls: i + 1
            for i, cls in enumerate(self.classes)
        }
        if dataset_scope in ['voc07', 'voc12']:
            if not test_mode:
                cache_file = osp.join(cache_dir, dataset_scope + '_train.json')
            else:
                cache_file = osp.join(cache_dir, dataset_scope + '_test.json')
            exist_or_mkdir(cache_file)
            img_prefix = osp.join(dataset_root, 'JPEGImages/')
            if not file_is_exist(cache_file):
                dataset_infos = self._parse_voc_single(
                    dataset_root, class_to_cat, test_mode)
                dump(dataset_infos, cache_file)
            return cache_file, img_prefix

        elif dataset_scope == 'voc07+12':
            if not test_mode:
                cache_file = osp.join(cache_dir, dataset_scope + '_train.json')
            else:
                cache_file = osp.join(cache_dir, dataset_scope + '_test.json')
            exist_or_mkdir(cache_file)
            img_prefix = dataset_root
            if not file_is_exist(cache_file):
                dataset_root07 = osp.join(dataset_root, 'voc2007/')
                dataset_root12 = osp.join(dataset_root, 'voc2012/')
                name_prefix07 = 'voc2007/JPEGImages/'
                name_prefix12 = 'voc2012/JPEGImages/'

                if test_mode:
                    dataset_infos = self._parse_voc_single(
                        dataset_root12, class_to_cat,
                        test_mode=test_mode, name_prefix=name_prefix12)
                else:
                    dataset_infos = []
                    dataset_infos_trainval07 = self._parse_voc_single(
                        dataset_root07, class_to_cat,
                        test_mode=False, name_prefix=name_prefix07)
                    dataset_infos.extend(dataset_infos_trainval07)
                    dataset_infos_test07 = self._parse_voc_single(
                        dataset_root07, class_to_cat,
                        test_mode=True, name_prefix=name_prefix07)
                    dataset_infos.extend(dataset_infos_test07)
                    dataset_infos_trainval12 = self._parse_voc_single(
                        dataset_root12, class_to_cat,
                        test_mode=test_mode, name_prefix=name_prefix12)
                    dataset_infos.extend(dataset_infos_trainval12)
                dump(dataset_infos, cache_file)
            return cache_file, img_prefix

    def _parse_voc_single(self, dataset_root, class_to_cat,
                          test_mode, name_prefix=''):
        ann_prefix = osp.join(dataset_root, 'annotations/')
        if not test_mode:
            filepath = osp.join(
                dataset_root, 'ImageSets/Main/trainval.txt')
        else:
            filepath = osp.join(
                dataset_root, 'ImageSets/Main/test.txt')
        with open(filepath, 'r') as f:
            lines = f.readlines()

        dataset_infos = []
        for line in lines:
            annotation_file = osp.join(
                ann_prefix, line.strip() + '.xml')
            data = self._parse_ann_info(
                annotation_file, class_to_cat, name_prefix)
            dataset_infos.append(data)
        return dataset_infos

    def _parse_ann_info(self, annotation_file, class_to_cat, name_prefix):
        img_info = ET.parse(annotation_file)
        img_name = name_prefix + img_info.find('filename').text.lower().strip()
        size = img_info.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        objs = img_info.findall('object')

        bboxes = []
        labels = []
        bboxes_ignore = []
        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                bboxes_ignore.append([x1, y1, x2, y2])
            else:
                cls_name = obj.find('name').text.lower().strip()
                bboxes.append([x1, y1, x2, y2])
                labels.append(class_to_cat[cls_name])
        ann = dict(
            bboxes=np.array(bbox, dtype=np.float32),
            labels=np.array(labels, dtype=np.int64),
            bboxes_ignore=np.array(bboxes_ignore, dtype=np.float32)
        )
        data = dict(
            filename=img_name,
            width=width,
            height=height,
            ann=ann)
        return data
