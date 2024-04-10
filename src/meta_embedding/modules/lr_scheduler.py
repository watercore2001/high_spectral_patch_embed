import math
import torch
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["CosineAnnealingWithWarmup"]


class CosineAnnealingWithWarmup(LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Linear warmup step size. Default: 0.
        annealing_epochs (int): annealing steps
        max_lr (float): First cycle's max learning rate.
        min_lr (float): Min learning rate.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 annealing_epochs: int,
                 max_lr: float,
                 min_lr: float,
                 ):

        self.warmup_epochs = warmup_epochs  # warmup epochs
        self.annealing_epochs = annealing_epochs
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.current_epoch = 0

        # will call self.step() in __init__
        super().__init__(optimizer)

    def get_lr(self):
        assert self.current_epoch >= 1

        if self.current_epoch <= self.warmup_epochs:
            lr = self.max_lr * self.current_epoch / self.warmup_epochs
        elif self.current_epoch <= self.warmup_epochs + self.annealing_epochs:
            # 0 -> 1
            annealing_phase = (self.current_epoch - self.warmup_epochs) / self.annealing_epochs
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * annealing_phase)) / 2
        else:
            lr = self.min_lr

        return [lr] * len(self.optimizer.param_groups)

    def step(self, epoch=None):
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
