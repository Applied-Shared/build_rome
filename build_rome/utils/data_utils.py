import math
from typing import *

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


def recursive_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = False,
) -> Any:
    """
    Recursively move all tensors in a data structure to a device.
    """
    if hasattr(data, "to"):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    else:
        return data


def load_balanced_group_indices(
    load: List[int],
    num_groups: int,
    equal_size: bool = False,
) -> List[List[int]]:
    """
    Split indices into groups with balanced load.
    """
    if equal_size:
        group_size = len(load) // num_groups
    indices = np.argsort(load)[::-1]
    groups = [[] for _ in range(num_groups)]
    group_load = np.zeros(num_groups)
    for idx in indices:
        min_group_idx = np.argmin(group_load)
        groups[min_group_idx].append(idx)
        if equal_size and len(groups[min_group_idx]) == group_size:
            group_load[min_group_idx] = float("inf")
        else:
            group_load[min_group_idx] += load[idx]
    return groups


class TokenBalancedDistributedBatchSampler:
    """Batch sampler that equalizes per-rank token loads at EVERY step.

    Plain DDP sampling gives each rank an independent random draw; with sparse
    SLat windows (a few hundred to ~20k+ tokens each) the per-rank batch sums
    differ by 10-30%, and every gradient all-reduce waits for the heaviest
    rank. Here all ranks deterministically shuffle the same permutation
    (seed + epoch), split it into pools of ``num_replicas * batch_size``
    samples, greedy-bin-pack each pool into ``num_replicas`` equal-size groups
    by load (``load_balanced_group_indices``), and rank r takes group r — so
    each step's per-rank loads are near-identical while every sample is still
    seen exactly once per epoch (up to the dropped tail pool).

    ``loads`` may be any per-sample cost proxy; the chunk datasets use the
    latent file's byte size (r > 0.999 against token count for npz latents).

    Requires ``Trainer(use_distributed_sampler=False)``: ranks are already
    disjoint by construction, Lightning must not re-wrap the sampler. The
    epoch advances automatically per ``__iter__`` (all ranks iterate in
    lockstep under DDP); ``set_epoch`` overrides it for checkpoint resume.
    """

    def __init__(
        self,
        loads: List[int],
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = True,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        assert 0 <= rank < num_replicas, (rank, num_replicas)
        self.loads = list(loads)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0
        self.pool_size = self.batch_size * self.num_replicas
        assert len(self.loads) >= self.pool_size, (
            f"need at least {self.pool_size} samples (batch {self.batch_size} x "
            f"{self.num_replicas} ranks), got {len(self.loads)}"
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.loads) // self.pool_size

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed * 1_000_003 + self._epoch)
            perm = torch.randperm(len(self.loads), generator=g).tolist()
        else:
            perm = list(range(len(self.loads)))
        for start in range(0, len(self) * self.pool_size, self.pool_size):
            pool = perm[start : start + self.pool_size]
            groups = load_balanced_group_indices(
                [self.loads[i] for i in pool], self.num_replicas, equal_size=True
            )
            yield [pool[j] for j in groups[self.rank]]
        # All ranks iterate in lockstep, so advancing here keeps shuffles
        # aligned even if the trainer never calls set_epoch.
        self._epoch += 1


def cycle(data_loader: DataLoader) -> Iterator:
    while True:
        for data in data_loader:
            if isinstance(data_loader.sampler, ResumableSampler):
                data_loader.sampler.idx += data_loader.batch_size  # type: ignore[attr-defined]
            yield data
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.epoch += 1
        if isinstance(data_loader.sampler, ResumableSampler):
            data_loader.sampler.epoch += 1
            data_loader.sampler.idx = 0


class ResumableSampler(Sampler):
    """
    Distributed sampler that is resumable.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.idx = 0
        self.drop_last = drop_last
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.world_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.world_size) / self.world_size  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]

        # resume from previous state
        indices = indices[self.idx :]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "idx": self.idx,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.idx = state_dict["idx"]


class BalancedResumableSampler(ResumableSampler):
    """
    Distributed sampler that is resumable and balances the load among the processes.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ) -> None:
        assert hasattr(dataset, "loads"), 'Dataset must have "loads" attribute to use BalancedResumableSampler'
        super().__init__(dataset, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.loads = dataset.loads

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # balance load among processes
        num_batches = len(indices) // (self.batch_size * self.world_size)
        balanced_indices = []
        for i in range(num_batches):
            start_idx = i * self.batch_size * self.world_size
            end_idx = (i + 1) * self.batch_size * self.world_size
            batch_indices = indices[start_idx:end_idx]
            batch_loads = [self.loads[idx] for idx in batch_indices]
            groups = load_balanced_group_indices(batch_loads, self.world_size, equal_size=True)
            balanced_indices.extend([batch_indices[j] for j in groups[self.rank]])

        # resume from previous state
        indices = balanced_indices[self.idx :]

        return iter(indices)
