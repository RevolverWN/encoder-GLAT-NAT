import argparse
import logging
from typing import Union, Dict, List

import torch
import numpy as np

from dataset.iterator import DataHandler
from tasks import register_task
from torch import Tensor

from tasks.trans_utils import assign_single_value_long, assign_multi_value_long
from tasks.translation import TranslationTask

logger = logging.getLogger(__name__)

register_name = "share-translation"

default_dict = {"src_lang": {"type": str, "help": "give the source language prefix, eg: en"},
                "tgt_lang": {"type": str, "help": "give the target language prefix, eg: de"},
                "init_mix_target_prob": {"type": float, "default": 0.5,
                                         "help": "The probability of using target tokens as "
                                                 "encoder input"},
                "anneal_mix_target_prob": {"type": bool, "default": True,
                                           "help": "Whether to anneal the probability of using"
                                                   "target tokens as encoder input"},
                "total_steps": {"type": int, "default": 0,
                                "help": "The total steps of the training stage"},
                "rank_chunks_multiplier": {"type": int, "default": 150,
                                           "help": "if total_steps is zero, we set total steps as a multiplier of "
                                                   "rank_chunks in an epoch"}
                }


@register_task(register_name)
class ShareTranslation(TranslationTask):
    config = default_dict

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def build_data_handler(self, split, chunk_size, max_tokens, max_sentences, rank, gpu_count, device):
        iterator = DataHandler(self.dataset[split], chunk_size, rank, device, gpu_count, max_tokens,
                               max_sentences)
        if split == "train":
            self.train_iterator = iterator
            if self.config.total_steps == 0:
                self.train_total_steps = float(iterator.rank_total_chunks * self.config.rank_chunks_multiplier)
            else:
                self.train_total_steps = float(self.config.total_steps)
        return iterator

    def step(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model, criterion, train_flag=False,
             eval_flag=False, inference_flag=False):
        """
        :param batch: batch is either a dict or an empty list due to multi-gpu last minibatch padding to same length
                      with empty list
        :param model:
        :param criterion:
        :return: outputs: outputs is dict type, all values in outputs should be synchronized, so we pack these values
                          into a dict.
                 logits: used for validation to get the best token.
        """

        # TODO: total updates based on chunk number, so we must know
        def train_step(batch, model, criterion):
            if self.config.anneal_mix_target_prob:
                mix_target_prob = self.config.init_mix_target_prob * max(
                    1. - self.train_iterator.rank_total_updates.double() / self.train_total_steps, 0.)
                mix_target_flag = np.random.sample() < mix_target_prob
                if mix_target_flag:
                    model.config.src_embedding_copy = False
                else:
                    model.config.src_embedding_copy = True

            model_outputs: dict = model(batch["net_input"]["src_tokens"],
                                        batch["net_input"]["src_masks"],
                                        batch["target"],
                                        batch["target_masks"])

            label_smoothing_loss, nll_loss = criterion(model_outputs, batch["target"], batch["target_masks"],
                                                       self.tgt_dict.padding_id)

            logging_outputs = {"label_smoothing_loss": label_smoothing_loss,
                               "nll_loss": nll_loss,
                               "nsentences": batch["nsentences"],
                               "ntokens": batch["ntokens"],
                               "valid_batch_flag": torch.tensor(1, dtype=torch.int64,
                                                                device=self.global_config.device)}

            logits = model_outputs["logits"]
            return logging_outputs, logits

        def inference_step(batch, model):
            if not hasattr(model, self.global_config.model_decoding_strategy):
                raise AttributeError(
                    "model have not {} decoding_strategy".format(self.global_config.model_decoding_strategy))

            decoding_strategy_method = getattr(model, self.global_config.model_decoding_strategy)

            model_outputs: dict = decoding_strategy_method(batch["net_input"]["src_tokens"],
                                                           batch["net_input"]["src_masks"])
            logging_outputs = {"nsentences": batch["nsentences"],
                               "ntokens": batch["ntokens"]}
            return logging_outputs, model_outputs["hypo_tokens"]

        assert sum(
            (train_flag, eval_flag, inference_flag)) == 1, "train_flag {}, eval_flag {}, inference_flag {} must have" \
                                                           " one and only one True value".format(train_flag, eval_flag,
                                                                                                 inference_flag)

        if batch:
            for key, value in batch.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        batch[key][inner_key] = inner_value.to(device=self.global_config.device)
                else:
                    batch[key] = value.to(device=self.global_config.device)

            if train_flag:
                model.train()
                self.train_timer.start()
                try:
                    logging_outputs, logits = train_step(batch, model, criterion)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging_outputs, logits = self.bulid_empty_return_value()
                        logger.warning("CUDA out of memory, cross this batch!")
                self.train_timer.pause()
                logging_outputs["time_interval"] = self.train_timer.current_time_interval
                return logging_outputs, logits

            elif eval_flag:
                model.eval()

                self.eval_timer.start()
                logging_outputs, _ = train_step(batch, model, criterion)

                torch.cuda.empty_cache()
                model_outputs, hypo_tokens = inference_step(batch, model)
                self.eval_timer.pause()

                logging_outputs["time_interval"] = self.eval_timer.current_time_interval

                return logging_outputs, hypo_tokens
            else:
                model.eval()
                self.inference_timer.start()
                logging_outputs, hypo_tokens = inference_step(batch, model)
                self.inference_timer.pause()
                logging_outputs["time_interval"] = self.inference_timer.current_time_interval
                return logging_outputs, hypo_tokens

        else:
            return self.bulid_empty_return_value()

    def bulid_empty_return_value(self):
        empty_batch_logging_outputs = {
            "label_smoothing_loss": torch.tensor(0, dtype=torch.float32, device=self.global_config.device),
            "nll_loss": torch.tensor(0, dtype=torch.float32, device=self.global_config.device),
            "nsentences": torch.tensor(0, dtype=torch.int64, device=self.global_config.device),
            "ntokens": torch.tensor(0, dtype=torch.int64, device=self.global_config.device),
            "valid_batch_flag": torch.tensor(0, dtype=torch.int64, device=self.global_config.device),
            "time_interval": torch.tensor(0, dtype=torch.float32, device=self.global_config.device)}

        empty_batch_logits = torch.tensor(0, dtype=torch.float32)
        return empty_batch_logging_outputs, empty_batch_logits
