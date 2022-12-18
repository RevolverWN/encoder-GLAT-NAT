import argparse
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from criterions import register_criterion
from loss_dropper import LossDropper
import torch.nn as nn


def mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


register_name = "label_smoothed_cross_entropy_sent_align"

default_dict = {
    "label_smoothing": {"type": float, "default": 0.1, "help": "label smoothing"},
    "length_loss": {"action": "store_true", "help": "length prediction"},
    "length_loss_factor": {"type": float, "default": 0.1, "help": "weights on the length prediction loss"},

    "feature_loss_factor": {"type": float, "default": 0.1, "help": "weights on the feature prediction loss"}
}


@register_criterion(register_name)
class LabelSmoothedCrossEntropyCriterionSentAlign(_Loss):
    config = default_dict

    def __init__(self, config: namedtuple, train_iterator=None):
        super(LabelSmoothedCrossEntropyCriterionSentAlign, self).__init__()
        self.config = config
        self.eps = torch.tensor(self.config.label_smoothing, dtype=torch.float32)
        self.feature_loss = nn.MSELoss()

        if train_iterator:
            self.train_iterator = train_iterator

        if self.config.length_loss:
            self.length_loss_fn = torch.nn.CrossEntropyLoss()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def forward(self, model_outputs: dict, targets, tgt_masks, padding_idx, reduce=True):
        logits = model_outputs["logits"]
        decoder_features = model_outputs["decoder_features"]
        length_logits = model_outputs["length_logits"]

        encoder_outputs = model_outputs["encoder_outputs"]
        src_masks = encoder_outputs["src_masks"]
        src_emb = encoder_outputs["src_emb"]
        encoder_features = encoder_outputs["encoder_features"]

        batch_size, seq_len, vocab_size = logits.shape

        # feature loss
        encoder_features = encoder_features.transpose(0, 1)
        decoder_features = decoder_features.transpose(0, 1)
        encoder_features_mean = mean_pooling(encoder_features, src_masks)
        decoder_features_mean = mean_pooling(decoder_features, tgt_masks)

        feature_loss = self.feature_loss(encoder_features_mean, decoder_features_mean)

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

        # TODO:length_loss (whose value is e.g. 5.4) is far less than smooth_loss and
        #  feature_loss is less than length_loss by an order of magnitude 
        if self.config.length_loss:
            length_loss = self.length_loss_fn(length_logits, (~tgt_masks).sum(1))
            loss = loss + self.config.length_loss_factor * length_loss + self.config.feature_loss_factor * feature_loss

        return loss, nll_loss
