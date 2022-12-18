import argparse
from collections import namedtuple

import torch
from lr_schedulers import register_scheduler
from lr_schedulers.lr_scheduler_base import LrSchedulerBase

register_name = "polynomial_decay_scheduler"

default_dict = {
    "init_lr": {"type": float, "default": 3e-4, "help": " "},
    "final_lr": {"type": float, "default": 1e-5, "help": "learning rate to decay to"},
    "total_steps": {"type": float, "default": 250000., "help": "total number of updates over which to decay learning rate"},
    "warmup_updates": {"type": int, "default": 0, "help":
        "warmup the learning rate linearly for the first N updates"},
    "force_anneal": {"default": None, "help": "force annealing at specified epoch"},
    "power": {"default": 1.0, "help": "decay exponent"},
}


@register_scheduler(register_name)
class PolynomialDecayLRSchedule(LrSchedulerBase):
    config = default_dict

    def __init__(self, optimizer: torch.optim.Optimizer, config: namedtuple):
        super().__init__()
        self.optimizer = optimizer
        self.config = config

        assert self.config.total_steps > 0

        self.init_lr = self.lr = self.config.init_lr
        if self.config.warmup_updates > 0:
            self.warmup_factor = 1.0 / self.config.warmup_updates
        else:
            self.warmup_factor = 1
        self.final_lr = self.config.final_lr
        self.total_steps = self.config.total_steps
        self.power = self.config.power
        self.optimizer.param_groups[0]['lr'] = self.warmup_factor * self.init_lr

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def get_next_lr(self, epoch):
        lrs = [self.config.init_lr]
        if self.config.force_anneal is None or epoch < self.config.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.optimizer.param_groups[0]['lr']
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.param_groups[0]['lr'] = self.warmup_factor * self.lr

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.config.warmup_updates > 0 and num_updates <= self.config.warmup_updates:
            self.warmup_factor = num_updates / float(self.config.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif num_updates >= self.total_steps:
            lr = self.final_lr
        else:
            warmup = self.config.warmup_updates
            lr_range = self.lr - self.final_lr
            pct_remaining = 1 - (num_updates - warmup) / (self.total_steps - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.final_lr

        self.optimizer.param_groups[0]['lr'] = lr

    def state_dict(self):
        pass

    def reset(self):
        pass