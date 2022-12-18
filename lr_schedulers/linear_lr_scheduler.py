import argparse
from collections import namedtuple

import torch
from lr_schedulers import register_scheduler
from lr_schedulers.lr_scheduler_base import LrSchedulerBase

register_name = "linear_lr_scheduler"

default_dict = {
    "init_lr": {"type": float, "default": 3e-4, "help": " "},
    "final_lr": {"type": float, "default": 1e-5, "help": "learning rate to decay to"},
    "total_steps": {"type": int, "default": 250000, "help": "total number of updates over which to decay learning rate"}
}


@register_scheduler(register_name)
class Linear_LR_Schedule(LrSchedulerBase):
    config = default_dict

    def __init__(self, optimizer: torch.optim.Optimizer, config: namedtuple):
        super().__init__()
        self.optimizer = optimizer
        self.config = config

        self.init_lr = self.config.init_lr
        self.final_lr = self.config.final_lr
        self.total_steps = self.config.total_steps
        self.slope = (self.init_lr - self.final_lr) / self.total_steps

        self.optimizer.param_groups[0]['lr'] = self.init_lr

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def step_update(self, num_updates):
        scale = 1.0 - num_updates * self.slope / self.init_lr
        scale = max(scale, 0.)

        lr = self.init_lr * scale

        self.optimizer.param_groups[0]['lr'] = lr

    def state_dict(self):
        pass

    def reset(self):
        pass