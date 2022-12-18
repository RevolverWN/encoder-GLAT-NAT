import argparse
from collections import namedtuple

import torch
from lr_schedulers import register_scheduler
from lr_schedulers.lr_scheduler_base import LrSchedulerBase

register_name = "transformer_lr_scheduler"

default_dict = {
    "init_lr": {"type": float, "default": 1., "help": " "},
    "model_size": {"type": int, "default": 512, "help": " "},
    "warmup_steps": {"type": int, "default": 4000, "help": "warmup the learning rate linearly for the first N updates"}
}


@register_scheduler(register_name)
class Transformer_LR_Schedule(LrSchedulerBase):
    config = default_dict

    def __init__(self, optimizer: torch.optim.Optimizer, config: namedtuple):
        super().__init__()
        self.optimizer = optimizer
        self.config = config

        self.init_lr = self.config.init_lr
        self.model_size = self.config.model_size
        self.warmup_steps = self.config.warmup_steps

        # self.optimizer.param_groups[0]['lr'] = self.init_lr

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def step_update(self, num_updates):
        num_updates += 1
        scale = self.model_size ** -0.5
        scale *= min(num_updates ** -0.5, num_updates * self.warmup_steps ** -1.5)

        lr = self.init_lr * scale

        self.optimizer.param_groups[0]['lr'] = lr

    def state_dict(self):
        pass

    def reset(self):
        pass
