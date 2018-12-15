import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from builder import obj_from_dict
import datasets


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but add an extra
    field `flag` to identify different group for the GroupSampler
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self._set_group_flag()

    def _set_group_flag(self):
        if hasattr(self.datasets[0], 'flag'):
            flags = []
            for dataset_idx in range(len(self.datasets)):
                flags.append(self.datasets[dataset_idx].flag)
            self.flag = np.concatenate(flags)


def get_datasets(dataset_cfg, parent=datasets, additional_dict=None):
    if isinstance(dataset_cfg['ann_file'], (tuple, list)):
        ann_files = dataset_cfg['ann_file']
        num_dataset = len(ann_files)
    else:
        ann_files = [dataset_cfg['ann_file']]
        num_dataset = 1

    if 'proposal_file' in dataset_cfg:
        if isinstance(dataset_cfg['proposal_file'], (tuple, list)):
            proposal_files = dataset_cfg['proposal_file']
        else:
            proposal_files = [dataset_cfg['proposal_file']] * num_dataset
    else:
        proposal_files = [None] * num_dataset
    assert len(proposal_files) == num_dataset

    if isinstance(dataset_cfg['img_prefix'], (tuple, list)):
        img_prefixes = dataset_cfg['img_prefix']
    else:
        img_prefixes = [dataset_cfg['img_prefix']] * num_dataset
    assert len(img_prefixes) == num_dataset

    datasets = []
    for i in range(num_dataset):
        args_dict = dataset_cfg.copy()
        args_dict['ann_file'] = ann_files[i]
        args_dict['proposal_file'] = proposal_files[i]
        args_dict['img_prefix'] = img_prefixes[i]
        dataset = obj_from_dict(args_dict, parent, additional_dict)
        datasets.append(dataset)
    if num_dataset > 1:
        datasets = ConcatDataset(datasets)
    else:
        datasets = datasets[0]
    return datasets
