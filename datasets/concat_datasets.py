import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


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
