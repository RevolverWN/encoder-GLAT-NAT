import logging
from typing import Union, Dict, List

import torch

from tasks import register_task
from torch import Tensor

from tasks.trans_utils import assign_single_value_long, assign_multi_value_long
from tasks.translation import TranslationTask

logger = logging.getLogger(__name__)

register_name = "semi-translation"

default_dict = {
    "src_lang": {"type": str, "help": "give the source language prefix, eg: en"},
    "tgt_lang": {"type": str, "help": "give the target language prefix, eg: de"}
}


@register_task(register_name)
class SemiTranslation(TranslationTask):
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

        def train_step(batch, model, criterion):
            # TODO semi-autoregressive generation, #tokens need to modify, no source embedding copy
            # TODO padding disappear! label smooth need to modify?
            # input_tgt_tokens = batch["target"]
            batch_size, seq_len = batch["target"].size()
            index_range = torch.arange(seq_len)[None, :].repeat((batch_size, 1))
            permutation = torch.randperm(seq_len)
            index_range = index_range[:, permutation]
            splits_tuple = torch.tensor_split(index_range, 4, dim=1)

            label_smoothing_loss_list = []
            nll_loss_list = []
            input_tgt_tokens = batch["target"].clone()
            for i, split in enumerate(splits_tuple):
                # input_tgt_tokens = batch["target"].clone()
                if i == 0:
                    mask_index = torch.cat(splits_tuple, dim=1)
                    input_tgt_tokens = assign_single_value_long(input_tgt_tokens, mask_index, self.tgt_dict.mask_id)
                    # del mask_index
                if i - 1 >= 0:
                    input_tgt_tokens = assign_multi_value_long(input_tgt_tokens, splits_tuple[i-1], batch["target"])
                input_tgt_tokens = input_tgt_tokens.masked_fill(batch["target_masks"], self.tgt_dict.padding_id)

                model_outputs: dict = model(batch["net_input"]["src_tokens"],
                                            batch["net_input"]["src_masks"],
                                            input_tgt_tokens,
                                            batch["target_masks"])

                token_masks = torch.cat([splits_tuple[j] for j in range(len(splits_tuple)) if j!=i], dim=1)
                label_smoothing_loss, nll_loss = criterion(model_outputs,
                                                           batch["target"],
                                                           batch["target_masks"],
                                                           self.tgt_dict.padding_id,
                                                           token_masks,
                                                           self.tgt_dict.mask_id)
                del token_masks
                label_smoothing_loss_list.append(label_smoothing_loss)
                nll_loss_list.append(nll_loss)

            # 1 indice split by 4
            # 2 first index, mask tgt_tokens by full index
            # 3 compute one index loss
            # 4 second index, mask tgt_tokens by the other three index
            # 1 - full, 2 - last3, 3 - last2, 4 - last3
            # model_outputs: dict = model(batch["net_input"]["src_tokens"],
            #                             batch["net_input"]["src_masks"],
            #                             batch["target"],
            #                             batch["target_masks"])
            #
            # label_smoothing_loss, nll_loss = criterion(model_outputs, batch["target"], batch["target_masks"],
            #                                            self.tgt_dict.padding_id)
            label_smoothing_loss = sum(label_smoothing_loss_list)
            nll_loss = sum(nll_loss_list)

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
                logging_outputs, logits = train_step(batch, model, criterion)
                self.train_timer.pause()
                logging_outputs["time_interval"] = self.train_timer.current_time_interval
                return logging_outputs, logits

            elif eval_flag:
                model.eval()

                self.eval_timer.start()
                logging_outputs, _ = train_step(batch, model, criterion)
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
            label_smoothing_loss = torch.tensor(0, dtype=torch.float32, device=self.global_config.device)
            nll_loss = torch.tensor(0, dtype=torch.float32, device=self.global_config.device)

            logging_outputs = {"label_smoothing_loss": label_smoothing_loss,
                               "nll_loss": nll_loss,
                               "nsentences": torch.tensor(0, dtype=torch.int64, device=self.global_config.device),
                               "ntokens": torch.tensor(0, dtype=torch.int64, device=self.global_config.device),
                               "valid_batch_flag": torch.tensor(0, dtype=torch.int64, device=self.global_config.device)}

            logits = torch.tensor(0, dtype=torch.float32)
            logging_outputs["time_interval"] = torch.tensor(0, dtype=torch.float32, device=self.global_config.device)
            return logging_outputs, logits
