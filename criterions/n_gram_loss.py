import argparse
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from criterions import register_criterion

register_name = "n_gram_loss"

default_dict = {
    "label_smoothing": {"type": float, "default": 0.1, "help": "label smoothing"},
    "length_loss": {"action": "store_true", "help": "length prediction"},
    "length_loss_factor": {"type": float, "default": 0.1, "help": "weights on the length prediction loss"},
    "n_gram": {"type": int, "default": 2, "help": "label smoothing"}
}


@register_criterion(register_name)
class NGramCriterion(_Loss):
    config = default_dict

    def __init__(self, config: namedtuple):
        super(NGramCriterion, self).__init__()
        self.config = config
        self.eps = torch.tensor(self.config.label_smoothing, dtype=torch.float32)
        if self.config.length_loss:
            self.length_loss_fn = torch.nn.CrossEntropyLoss()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def forward(self, model_outputs: dict, targets, tgt_masks, padding_idx, reduce=True):
        logits = model_outputs["logits"]
        batch_size, seq_len, _ = logits.size()
        n_gram_loss_list = []

        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target_flat = targets.view(-1)
        nll_loss = -torch.gather(input=lprobs, dim=-1, index=target_flat.unsqueeze(-1)).squeeze()
        if padding_idx:
            nll_loss.masked_fill_(mask=(target_flat == padding_idx), value=0.0)

        for i in range(1, self.config.n_gram+1):
            if i == 1:
                smooth_loss = -lprobs.sum(dim=-1)
                if padding_idx:
                    smooth_loss.masked_fill_(mask=(target_flat == padding_idx), value=0.0)
                if reduce:
                    one_gram_nll_loss = nll_loss.sum()
                    smooth_loss = smooth_loss.sum()
                eps_i = self.eps / (lprobs.size(-1) - 1)
                one_gram_loss = (1.0 - self.eps - eps_i) * one_gram_nll_loss + eps_i * smooth_loss
                n_gram_loss_list.append(one_gram_loss)
            if i >= 2:
                nll_loss = nll_loss.view(batch_size, seq_len)
                nll_loss_list = [nll_loss[:, j: seq_len-(i-j-1)] for j in range(i)]
                gram_loss = nll_loss_list[0]
                for k in nll_loss_list[1:]:
                    gram_loss = gram_loss + k
                n_gram_loss_list.append(gram_loss.sum())

        loss = sum(n_gram_loss_list)

        if self.config.length_loss:
            length_logits = model_outputs["length_logits"]
            length_loss = self.length_loss_fn(length_logits, (~tgt_masks).sum(1))
            loss = loss + self.config.length_loss_factor * length_loss

        return loss, nll_loss.sum()