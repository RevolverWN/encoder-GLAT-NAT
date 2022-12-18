import argparse
import logging
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import nn

from criterions import register_criterion
from loss_dropper import LossDropper

from tasks.trans_utils import assign_single_value_long

logger = logging.getLogger(__name__)

register_name = "label_smoothed_cross_entropy_KL_loss"

default_dict = {
    "label_smoothing": {"type": float, "default": 0.1, "help": "label smoothing"},
    "length_loss": {"action": "store_true", "help": "length prediction"},
    "length_loss_factor": {"type": float, "default": 0.1, "help": "weights on the length prediction loss"},
    "kl_loss": {"action": "store_true", "help": ""},
    "kl_loss_factor": {"type": float, "default": 1., "help": "weights on the source and target embedding KL loss"},
    "temperature": {"type": float, "default": 1.0, "help": "softmax temperature"}
}


def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


@register_criterion(register_name)
class LabelSmoothedCrossEntropyKLCriterion(_Loss):
    config = default_dict

    def __init__(self, config: namedtuple, train_iterator=None):
        super(LabelSmoothedCrossEntropyKLCriterion, self).__init__()
        self.config = config
        self.eps = torch.tensor(self.config.label_smoothing, dtype=torch.float32)

        self.kl_loss = nn.CosineEmbeddingLoss(reduction='mean')
        if train_iterator:
            self.train_iterator = train_iterator

        if self.config.length_loss:
            self.length_loss_fn = torch.nn.CrossEntropyLoss()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def forward(self, model_outputs: dict, targets, tgt_padding_masks, padding_idx, token_masks=None, mask_idx=None, reduce=True):

        logits = model_outputs["logits"]

        # token_ave_loss = torch.tensor(0, dtype=torch.float32, device=logits.device)
        sentence_ave_loss = torch.tensor(0, dtype=torch.float32, device=logits.device)
        kl_loss = torch.tensor(0, dtype=torch.float32, device=logits.device)

        batch_size, seq_len, vocab_size = logits.shape

        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target_flat = targets.view(-1)
        nll_loss = -torch.gather(input=lprobs, dim=-1, index=target_flat.unsqueeze(-1)).squeeze()
        smooth_loss = -lprobs.sum(dim=-1)
        if padding_idx:
            nll_loss.masked_fill_(mask=(target_flat == padding_idx), value=0.0)
            smooth_loss.masked_fill_(mask=(target_flat == padding_idx), value=0.0)

        if mask_idx:
            smooth_loss = smooth_loss.view(batch_size, seq_len)
            nll_loss = nll_loss.view(batch_size, seq_len)
            smooth_loss = assign_single_value_long(smooth_loss, token_masks, 0.0)
            nll_loss = assign_single_value_long(nll_loss, token_masks, 0.0)
            smooth_loss = smooth_loss.view(-1)
            nll_loss = nll_loss.view(-1)

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / (lprobs.size(-1) - 1)
        loss = (1.0 - self.eps - eps_i) * nll_loss + eps_i * smooth_loss

        if self.config.length_loss:
            length_logits = model_outputs["length_logits"]
            length_loss = self.length_loss_fn(length_logits, (~tgt_padding_masks).sum(1))
            loss = loss + self.config.length_loss_factor * length_loss

        if self.config.kl_loss:
            # tgt_emb = F.log_softmax(model_outputs["tgt_emb"] / self.config.temperature, dim=-1)
            # true_tgt_embed = F.softmax(model_outputs["true_tgt_embed"] / self.config.temperature, dim=-1)
            # kl_loss = self.config.kl_loss_factor * self.kl_loss(model_outputs["tgt_emb"], model_outputs["true_tgt_embed"])

            # kl_loss = (model_outputs["tgt_token_embed"] - model_outputs["true_tgt_embed"]) ** 2
            # kl_loss = kl_loss * (~tgt_padding_masks).unsqueeze(-1)
            # kl_loss = self.config.kl_loss_factor * torch.sqrt(kl_loss.sum()) / ((~tgt_padding_masks).sum() * model_outputs["tgt_emb"].size(2))
            loss_target = torch.ones_like(tgt_padding_masks, dtype=torch.int64)
            loss_target = loss_target.view(-1)
            tgt_token_embed = model_outputs["tgt_token_embed"].view(batch_size * seq_len, -1)
            true_tgt_embed = model_outputs["true_tgt_embed"].view(batch_size * seq_len, -1)
            kl_loss = self.kl_loss(tgt_token_embed, true_tgt_embed, loss_target)

        token_ave_loss = loss

        return token_ave_loss, kl_loss
