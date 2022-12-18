import argparse
from collections import namedtuple
from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from criterions import register_criterion

from torch import Tensor


register_name = "length_criterion"

default_dict = {
    "length_loss_factor": {"type": float, "default": 1, "help": "weights on the length prediction loss"}
}


@register_criterion(register_name)
class LengthCriterion(_Loss):
    config = default_dict
    samples_reduce = "nsentences"

    def __init__(self, config: namedtuple):
        super(LengthCriterion, self).__init__()
        self.config = config
        self.length_loss_fn = torch.nn.CrossEntropyLoss()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def forward(self, length_logits, targets: Tensor):
        loss = self.config.length_loss_factor * self.length_loss_fn(length_logits, targets)

        return loss