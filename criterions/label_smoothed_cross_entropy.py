import argparse
from collections import namedtuple
from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from criterions import register_criterion

from torch import Tensor


register_name = "label_smoothed_cross_entropy"

default_dict = {
    "label_smoothing": {"type": float, "default": 0.1, "help": "label smoothing"}}


@register_criterion(register_name)
class LabelSmoothedCrossEntropyCriterion(_Loss):
    config = default_dict
    samples_reduce = "ntokens"

    def __init__(self, config: namedtuple):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.config = config
        self.eps = torch.tensor(self.config.label_smoothing, dtype=torch.float32)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def forward(self,
                logits,
                targets: Tensor,
                padding_idx: Union[Tensor, int],
                reduce: bool = True):

        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target_flat = targets.view(-1)
        nll_loss = -torch.gather(input=lprobs, dim=-1, index=target_flat.unsqueeze(-1)).squeeze()
        smooth_loss = -lprobs.sum(dim=-1)
        if padding_idx:
            nll_loss.masked_fill_(mask=(target_flat == padding_idx), value=0.0)
            smooth_loss.masked_fill_(mask=(target_flat == padding_idx), value=0.0)

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / (lprobs.size(-1) - 1)
        loss = (1.0 - self.eps - eps_i) * nll_loss + eps_i * smooth_loss

        return loss


