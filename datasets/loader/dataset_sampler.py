from __future__ import division

import torch
import numpy as np

from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class GroupSampler(Sampler):
    """
    The function of this sampler is to sample images from dataset. It can work
    in the following two situations:
    1. during training, we sample sample_per_gpu * num_gpu images each time,
       and each group of sample_per_gpu images will be send to different gpu
       in the Parallel process, the image sizes between groups are different,
       and the flag in each group has to be the same, because we will pad the
       image in the next process.
    2. during single gpu testing, usually we only use sample_per_gpu=1 in
       testing to avoid padding images.
    """

    def __init__(self, dataset, sample_per_gpu=1):
        self.test_mode = dataset.test_mode
        if self.test_mode:
            assert sample_per_gpu == 1
            self.num_samples = len(dataset)
        else:
            assert hasattr(dataset, 'flag')
            self.flag = dataset.flag.astype(np.int64)
            self.group_sizes = np.bincount(self.flag)
            self.num_samples = 0
            for i, size in enumerate(self.group_sizes):
                self.num_samples += int(np.ceil(
                    size * 1.0 / sample_per_gpu)) * sample_per_gpu
        self.dataset = dataset
        self.sample_per_gpu = sample_per_gpu

    def __iter__(self):
        indices = []
        if self.test_mode:
            indices = range(len(self.dataset))
            return iter(indices)
        else:
            for i, size in enumerate(self.group_sizes):
                if size == 0:
                    continue
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                np.random.shuffle(indice)
                num_extra = int(np.ceil(size / self.sample_per_gpu)
                                ) * self.sample_per_gpu - len(indice)
                indice = np.concatenate([indice, indice[:num_extra]])
                indices.append(indice)
            indices = np.concatenate(indices)
            indices = [
                indices[i * self.sample_per_gpu:(i + 1) * self.sample_per_gpu]
                for i in np.random.permutation(
                    range(len(indices) // self.sample_per_gpu))
            ]
            indices = np.concatenate(indices)
            assert len(indices) == self.num_samples
            return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """
        Sampler that restricts data loading to a subset of the dataset.

        It is especially useful in conjunction with
        :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
        process can pass a DistributedSampler instance as a DataLoader sampler,
        and load a subset of the original dataset that is exclusive to it.

        .. note::
            Dataset is assumed to be of constant size.

        Arguments:
            dataset: Dataset used for sampling.
            sample_per_gpu: number of sample for each gpu.
            num_replicas (optional): Number of processes participating in
                distributed training.
            rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 sample_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()

        self.test_mode = dataset.test_mode
        if self.test_mode:
            assert sample_per_gpu == 1
            self.num_samples = int(
                np.ceil(len(dataset) * 1.0 / num_replicas))
        else:
            assert hasattr(dataset, 'flag')
            self.flag = dataset.flag.astype(np.int64)
            self.group_sizes = np.bincount(self.flag)

            self.num_samples = 0
            for i, size in enumerate(self.group_sizes):
                self.num_samples += int(
                    np.ceil(size * 1.0 / sample_per_gpu /
                            num_replicas)) * sample_per_gpu

        self.dataset = dataset
        self.sample_per_gpu = sample_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.test_mode:
            indices = range(len(self.dataset))
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()

            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
        else:
            for i, size in enumerate(self.group_sizes):
                if size == 0:
                    continue
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[torch.randperm(int(size),
                                               generator=g).tolist()].tolist()
                extra = int(
                    np.ceil(
                        size * 1.0 / self.sample_per_gpu / self.num_replicas)
                ) * self.sample_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice
            assert len(indices) == self.total_size

            indices = [indices[j]
                       for i in torch.randperm(
                len(indices) // self.sample_per_gpu, generator=g).tolist()
                for j in range(
                i * self.sample_per_gpu, (i + 1) * self.sample_per_gpu)
            ]

            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]
            assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
