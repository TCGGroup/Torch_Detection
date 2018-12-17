from functools import partial

from torch.utils.data import DataLoader

from .dataset_sampler import GroupSampler, DistributedGroupSampler
from .collate import collate

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     sample_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     world_size=1,
                     rank=0,
                     **kwargs):
    """
    To build data loader in a function way, this function can handle in both
    distributed situation and parallel situation, btw, `world_size` and `rank`
    parameters only used in distributed situation.
    """
    if dist:
        sampler = DistributedGroupSampler(
            dataset, sample_per_gpu, world_size, rank)
        batch_size = sample_per_gpu
        num_workers = workers_per_gpu
    else:
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = GroupSampler(dataset, sample_per_gpu)
        batch_size = num_gpus * sample_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=partial(collate, sample_per_gpu),
                             pin_memory=False,
                             **kwargs)
    return data_loader
