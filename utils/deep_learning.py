import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import torch.distributed as dist
import torch.multiprocessing as tmp
from functools import partial
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader as DataLoader


# 设置随机种子
def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 初始化分布式训练的环境
def init_dist(local_rank=0, backend='nccl'):
    tmp.set_start_method('spawn')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)


# 获取分布式训练的信息
def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


# ==========
# Dataloader
# ==========
class DistSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch.

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(
            self, dataset, num_replicas=None, rank=None, ratio=1
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # enlarged by ratio, and then divided by num_replicas
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on ite_epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        # enlarge indices
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # ==========subsample
        # e.g., self.rank=1, self.total_size=4, self.num_replicas=2
        # indices = indices[1:4:2] = indices[i for i in [1, 3]]
        # for the other worker, indices = indices[i for i in [0, 2]]
        # ==========
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """For shuffling data at each epoch. See train.py."""
        self.epoch = epoch


def create_dataloader(dataset, opts_dict, sampler=None, phase='train', seed=None):
    """Create dataloader."""
    if phase == 'train':
        # >I don't know why BasicSR have to detect `is_dist`
        dataloader_args = dict(
            dataset=dataset,
            batch_size=opts_dict['dataset']['train']['batch_size_per_gpu'],
            shuffle=False,  # sampler will shuffle at train.py
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'],
            sampler=sampler,
            drop_last=True,
            pin_memory=True
        )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            _worker_init_fn,
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'],
            rank=opts_dict['train']['rank'],
            seed=seed
        )

    elif phase == 'val':
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    return DataLoader(**dataloader_args)


def _worker_init_fn(worker_id, num_workers, rank, seed):
    # func for torch.utils.data.DataLoader
    # set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========
# Loss & Metrics
# ==========
# 计算图像或信号之间的差异时常用的损失函数，它在平方损失的基础上引入了平方根以提高对异常值的鲁棒性。
class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class FFTCharbonnierLoss(torch.nn.Module):
    def __init__(self, lambda_param=1.0, eps=1e-6):
        super(FFTCharbonnierLoss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss(eps=eps)
        self.lambda_param = lambda_param

    def forward(self, prediction, gt):
        fft_pred = fft.fft2(prediction, dim=(-2, -1))
        fft_gt = fft.fft2(gt, dim=(-2, -1))

        # Calculate amplitude and phase
        amp_pred = torch.sqrt(fft_pred.real**2 + fft_pred.imag**2)
        phase_pred = torch.atan2(fft_pred.imag, fft_pred.real)
        amp_gt = torch.sqrt(fft_gt.real**2 + fft_gt.imag**2)
        phase_gt = torch.atan2(fft_gt.imag, fft_gt.real)


        # Calculate Charbonnier Loss on amplitude and phase
        loss_amp = torch.mean(self.charbonnier_loss(amp_pred, amp_gt))
        loss_phase = torch.mean(self.charbonnier_loss(phase_pred, phase_gt))

        # Combine amplitude and phase losses
        total_loss = loss_amp + self.lambda_param * loss_phase

        return total_loss


class FFTL1Loss(torch.nn.Module):
    def __init__(self, lambda_param=1.0):
        super(FFTL1Loss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, pred, gt):
        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_gt = torch.fft.fft2(gt, dim=(-2, -1))

        amp_pred = torch.sqrt(fft_pred.real**2 + fft_pred.imag**2)
        phase_pred = torch.atan2(fft_pred.imag, fft_pred.real)

        amp_gt = torch.sqrt(fft_gt.real**2 + fft_gt.imag**2)
        phase_gt = torch.atan2(fft_gt.imag, fft_gt.real)

        # Calculate L1 Loss on amplitude and phase
        loss_amp = torch.mean(torch.abs(amp_pred - amp_gt))
        loss_phase = torch.mean(torch.abs(phase_pred - phase_gt))

        # Combine amplitude and phase losses
        total_loss = loss_amp + self.lambda_param * loss_phase

        return total_loss

class FFTLoss(torch.nn.Module):   #  FFT 损失函数  基于 L1 的
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.L1 = nn.L1Loss(reduction='mean')
    def forward(self, img1, img2):
        img1=torch.stack([torch.fft.fft2(img1, dim=(-2, -1)).real, torch.fft.fft2(img1, dim=(-2, -1)).imag], -1)
        img2=torch.stack([torch.fft.fft2(img2, dim=(-2, -1)).real, torch.fft.fft2(img2, dim=(-2, -1)).imag], -1)
        return self.L1(img1,img2)

class PSNR(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(PSNR, self).__init__()
        self.eps = eps
        self.mse_func = nn.MSELoss()

    def forward(self, X, Y):
        mse = self.mse_func(X, Y)
        psnr = 10 * math.log10(1 / (mse.item() + self.eps))
        return psnr


# ==========
# Scheduler
# ==========
import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ 这是一个带有重新启动机制的多步学习率方案的类，用于 PyTorch 中的优化器。以下是其主要参数和功能：
        optimizer（torch.nn.optimizer）：PyTorch 优化器的实例，它将根据学习率方案进行更新。
        milestones（list）：会降低学习率的迭代步骤。
        gamma（float）：学习率降低的比例，默认为 0.1。
        restarts（list）：重新启动的迭代步骤，默认为 [0]，表示在第0个迭代时重新启动。
        restart_weights（list）：在每次重新启动迭代步骤时的权重，默认为 [1]。
        last_epoch（int）：在 _LRScheduler 中使用的参数，默认为 -1。
    """

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=[0],
                 restart_weights=[1],
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


def get_position_from_periods(iteration, cumulative_period):
    """从周期列表中获取位置。
    它将返回周期列表中最接近右侧的数字的索引。
    例如，cumulative_period = [100, 200, 300, 400]，
    如果 iteration == 50，则返回 0；
    如果 iteration == 210，则返回 2；
    如果 iteration == 300，则返回 2。
    参数：
        iteration (int)：当前迭代次数。
        cumulative_period (list[int])：累积周期列表。
    返回：
        int：在周期列表中最接近右侧的数字的位置。
    """

    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
        余弦退火与重启的学习率方案。
        一个配置的示例：
        periods = [10, 10, 10, 10]
        restart_weights = [1, 0.5, 0.5, 0.5]
        eta_min=1e-7
        它有四个周期，每个周期有10次迭代。在第10、20、30次迭代时，调度器将使用restart_weights重新启动。
        参数：
            optimizer (torch.nn.optimizer)：PyTorch优化器。
            periods (list)：每个余弦退火周期的迭代次数。
            restart_weights (list)：在每次重启迭代时的重启权重。默认值：[1]。
            eta_min (float)：最小学习率。默认值：0。
            last_epoch (int)：在_LRScheduler中使用的参数。默认值：-1。
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=[1],
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(
            self.restart_weights)), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
