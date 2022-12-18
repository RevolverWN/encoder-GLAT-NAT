import argparse
from collections import namedtuple

import torch
from lr_schedulers import register_scheduler
from lr_schedulers.lr_scheduler_base import LrSchedulerBase

register_name = "inverse_square_root"

default_dict = {
    "warmup_init_lr": {"type": float, "default": 1e-07, "help": " "},
    "warmup_num_updates": {"type": int, "default": 4000, "help": "warmup the learning rate linearly for the first N updates"}
}


@register_scheduler(register_name)
class InverseSquareRootScheduler(LrSchedulerBase):
    config = default_dict

    def __init__(self, optimizer: torch.optim.Optimizer, config: namedtuple):
        super().__init__()
        self.optimizer = optimizer
        self.config = config
        self.warmup_init_lr = self.config.warmup_init_lr
        self.warmup_num_updates = self.config.warmup_num_updates

        warmup_end_lr = optimizer.param_groups[0]['lr']
        self.warmup_step = (warmup_end_lr - self.warmup_init_lr) / self.warmup_num_updates
        self.decay_factor = warmup_end_lr * self.config.warmup_num_updates ** 0.5

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def step_update(self, num_updates):
        if num_updates < self.warmup_num_updates:
            lr = self.warmup_init_lr + self.warmup_step * num_updates
        else:
            lr = self.decay_factor * num_updates ** -0.5

        self.optimizer.param_groups[0]['lr'] = lr
        return lr

    def state_dict(self):
        pass

    def reset(self):
        pass



