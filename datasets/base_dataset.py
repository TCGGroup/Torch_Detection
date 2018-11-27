from torch.utils.data import Dataset
import numpy as np
from .dataset_transforms import ImageTransforms, BboxTransforms, \
    MaskTransforms, BackgroundErasing
import os.path as osp
from .utils import load, is_list_of, to_tensor, random_scale, DataContainer


class BaseDataset(Dataset):
    """
    The data format in the annotation json file is as follow:

    [
        {
            'filename': 'a.jpg',
            'width': 600,
            'height': 1000,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4)
            }
        },
        ...
    ]
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_means,
                 img_stds,
                 img_expected_sizes,
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
                 debug=False):
        """
        Base Dataset class for detection, first convert the annotation file to
        the data format define above, then inherit the class from this class,
        and re-write the functions if needed.
        The order of the arguments match the order of transforms.

        Args:
            ann_file (str(json)): the file path where the annotation file is.
            img_prefix (str): the prefix of image path.
            img_means (tuple[float]): mean value for each channel of image.
            img_stds (tuple[float]): std value for each channel of image.
            img_expected_sizes (tuple or list[tuple]): the data format of
                scales is `(long_edge, short_edge)` or `[(long1, short1),
                (long2, short2), (long3, short3),...]`
            size_divisor (int): the number that the padded image can be
                divided by `size_divisor`
            flip_ratio (float): the ratio that we use flip in data augmentation
            be_cell_size (int): Background Erasing cell size, which used in the
                background erasing process.
            be_random_ratio (float): The ratio to drop background cells in the
                background erasing process.
            proposal_file (str(pkl)): the file path where the proposal file is.
            num_max_proposals (int): the maximum proposals we used in the model
            with_mask (bool): we use `mask` or not in the model
            with_crowd (bool): we use `crowd` object in training or not
            with_label (bool): we use ground truth label to supervise the model
                training or not, in `RPN`, we do not need `labels`, and when we
                are in the `test dataset`, we also do not need `labels`
            test_mode (bool): we are in the training or testing mode.
            with_background_erasing (bool): use background erasing or not.
            debug (bool): we are in the debug mode or not, if we are in the
                debug mode, we can show the annotations to check if we have
                parsed the annotation file rightly, and we can choose a small
                batch of image to debug.
        """
        # load annotations from annotation file
        self.img_infos = self.load_annotations(ann_file)
        # prefix of image path
        self.img_prefix = img_prefix
        # image normalization parameters
        self.img_means, self.img_stds = img_means, img_stds
        # img_expected_sizes list
        self.img_expected_sizes = img_expected_sizes \
            if isinstance(img_expected_sizes, list) else [img_expected_sizes]
        assert is_list_of(self.img_expected_sizes, tuple)
        # size divisor
        self.size_divisor = size_divisor
        # flip ratio
        self.flip_ratio = flip_ratio
        # background erasing settings
        self.be_cell_size = be_cell_size
        self.be_random_ratio = be_random_ratio

        # load the pkl proposal file
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        self.num_max_proposals = num_max_proposals

        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # with mask or not
        self.with_mask = with_mask
        # with crowd object or not
        self.with_crowd = with_crowd
        # training with label or not
        self.with_label = with_label
        # test mode
        self.test_mode = test_mode
        # with background erasing or not
        self.with_background_erasing = with_background_erasing
        # debug mode
        self.debug = debug
        if self.debug:
            self.img_infos = self.img_infos[:50]

        # set aspect ratio flag for images
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        self.img_transforms = ImageTransforms(img_means=self.img_means,
                                              img_stds=self.img_stds,
                                              size_divisor=self.size_divisor)
        self.bbox_transforms = BboxTransforms()
        self.mask_transforms = MaskTransforms()
        self.background_erasing = BackgroundErasing()

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return load(ann_file)

    def load_proposals(self, proposal_file):
        proposals = load(proposal_file)
        return proposals

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """
        Set aspect ratio flag for image.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self.img_infos), dtype=np.uint8)
        for i, img_info in enumerate(self.img_infos):
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            while True:
                data = self.prepare_train_img(idx)
                if data is None:
                    idx = self._rand_another(idx)
                    continue
                return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img_path = osp.join(self.img_prefix, img_info['filename'])

        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not proposal.shape[1] == 4 or proposal.shape[1] == 5:
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        if self.with_background_erasing:
            ann = self.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
        else:
            gt_bboxes = None

        def prepare_single_scale(img_path, expected_size, flip_ratio=0,
                                 proposal=None, bbox=None):
            _img, img_shape, pad_shape, scale_factor, \
                flipped_flag, flipped_direction = self.img_transforms(
                    img_path, expected_size, flip_ratio=flip_ratio)
            if bbox is not None:
                if not len(bbox) == 0:
                    _gt_bboxes = self.bbox_transforms(bbox,
                                                      img_shape,
                                                      scale_factor,
                                                      flipped_flag,
                                                      flipped_direction)
                else:
                    _gt_bboxes = bbox
                _img = self.background_erasing(
                    _img, img_shape, _gt_bboxes,
                    cell_size=self.be_cell_size,
                    random_ratio=self.be_random_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flipped_flag=flipped_flag,
                flipped_direction=flipped_direction
            )
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transforms(proposal,
                                                 img_shape,
                                                 scale_factor,
                                                 flipped_flag,
                                                 flipped_direction)
                _proposal = np.hstack([_proposal, score]) \
                    if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for expected_size in self.img_expected_sizes:
            # at first, we do not flip the image
            _img, _img_meta, _proposal = prepare_single_scale(
                img_path, expected_size, flip_ratio=0,
                proposal=proposal, bbox=gt_bboxes)
            imgs.append(_img)
            img_metas.append(DataContainer(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single_scale(
                    img_path, expected_size, flip_ratio=1,
                    proposal=proposal, bbox=gt_bboxes)
                imgs.append(_img)
                img_metas.append(DataContainer(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

    def prepare_train_img(self, idx):
        """
        Prepare an image for training, random select a scale from
        img_expected_sizes, and random flipping according to the probability.
        """
        img_info = self.img_infos[idx]
        img_path = osp.join(self.img_prefix, img_info['filename'])

        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            if len(proposals) == 0:
                return None
            if not proposals.shape[1] == 4 or proposals.shape[1] == 5:
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        # parse annotation for image
        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_bboxes_ignore = ann['bboxes_ignore']
        if len(gt_bboxes) == 0:
            return None

        # random select the training size for image
        expected_size = random_scale(self.img_expected_sizes)
        img, img_shape, pad_shape, \
            scale_factor, flipped_flag, flipped_direction = \
            self.img_transforms(img_path,
                                expected_size=expected_size,
                                flip_ratio=self.flip_ratio)
        # transform for proposals and bboxes
        if self.proposals is not None:
            proposals = self.bbox_transforms(proposals,
                                             img_shape,
                                             scale_factor,
                                             flipped_flag,
                                             flipped_direction)
            proposals = np.hstack([proposals, scores]) \
                if scores is not None else proposals
        gt_bboxes = self.bbox_transforms(gt_bboxes,
                                         img_shape,
                                         scale_factor,
                                         flipped_flag,
                                         flipped_direction)
        if self.with_background_erasing:
            img = self.background_erasing(img, img_shape, gt_bboxes,
                                          cell_size=self.be_cell_size,
                                          random_ratio=self.be_random_ratio)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transforms(gt_bboxes_ignore,
                                                    img_shape,
                                                    scale_factor,
                                                    flipped_flag,
                                                    flipped_direction)
        if self.with_mask:
            gt_masks = self.mask_transforms(ann['masks'],
                                            scale_factor,
                                            pad_shape,
                                            flipped_flag,
                                            flipped_direction)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flipped_flag=flipped_flag,
            flipped_direction=flipped_direction
        )

        data = dict(
            img=DataContainer(to_tensor(img), stack=True),
            img_meta=DataContainer(img_meta, cpu_only=True),
            gt_bboxes=DataContainer(to_tensor(gt_bboxes))
        )
        if self.proposals is not None:
            data['proposals'] = DataContainer(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DataContainer(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DataContainer(
                to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DataContainer(gt_masks, cpu_only=True)
        return data
