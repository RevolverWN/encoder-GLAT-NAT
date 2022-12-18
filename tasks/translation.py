import argparse
import logging
import os
from collections import namedtuple
from typing import List, Union, Dict, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

import criterions
import hooks
import metrics
import models
from dataset.dataset import IndexDataset, PairDataset
from dataset.iterator import DataHandler
from dictionary import Dictionary
from distribute import DistributedFairseqModel, get_global_group
from hooks.hook_base import HookList
from metrics.metric_base import MetricList
from tasks import register_task
from tasks.task_base import TaskBase

logger = logging.getLogger(__name__)

register_name = "translation"

default_dict: Dict[str, Dict] = {
    "src_lang": {"type": str, "help": "give the source language prefix, eg: en"},
    "tgt_lang": {"type": str, "help": "give the target language prefix, eg: de"}
}


@register_task(register_name)
class TranslationTask(TaskBase):
    config = default_dict
    default_metric = "bleu"

    def __init__(self, global_config: namedtuple, task_config: namedtuple):
        self.config = task_config
        self.global_config = global_config

        self.early_stop = False
        self.src_lang = task_config.src_lang
        self.tgt_lang = task_config.tgt_lang

        super(TranslationTask, self).__init__(global_config.data_dir)

        src_dict_path = os.path.join(self.data_dir, "dict." + self.src_lang + ".txt")
        tgt_dict_path = os.path.join(self.data_dir, "dict." + self.tgt_lang + ".txt")
        self.src_dict = Dictionary.load_dictionary_from_file(src_dict_path)
        self.tgt_dict = Dictionary.load_dictionary_from_file(tgt_dict_path)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def build_data_handler(self, split, chunk_size, max_tokens, max_sentences, rank, gpu_count, device, buffer_size):
        if split == self.global_config.train_prefix:
            self.train_data_handler = DataHandler(self.dataset[split], chunk_size, rank, device, gpu_count, max_tokens,
                                                  max_sentences, buffer_size)
            return self.train_data_handler
        elif split == self.global_config.valid_prefix:
            self.dev_data_handler = DataHandler(self.dataset[split], chunk_size, rank, device, gpu_count, max_tokens,
                                                max_sentences, buffer_size)
            return self.dev_data_handler
        elif split == self.global_config.test_prefix:
            self.test_data_handler = DataHandler(self.dataset[split], chunk_size, rank, device, gpu_count, max_tokens,
                                                 max_sentences, buffer_size)
            return self.test_data_handler

    def load_dataset(self, split: str) -> None:
        src_path = os.path.join(self.data_dir, split + "." + self.src_lang)
        tgt_path = os.path.join(self.data_dir, split + "." + self.tgt_lang)
        src_dataset = IndexDataset(src_path, self.src_dict)
        tgt_dataset = IndexDataset(tgt_path, self.tgt_dict)
        self.dataset[split] = PairDataset(src_dataset, tgt_dataset)

    def build_model(self, model_name: str):
        model_cls = models.registry[model_name]["cls"]
        config = models.registry[model_name]["default_config_dict"]
        self.model = model_cls(config, self.src_dict, self.tgt_dict)
        self.model.to(device=self.global_config.device)

        if self.global_config.gpu_count > 1:
            self._wrapped_model = DistributedFairseqModel(
                self.model,
                process_group=get_global_group(),
                device=self.global_config.rank,
            )
        else:
            self._wrapped_model = self.model

        return self._wrapped_model

    def build_criterion(self, criterion_name):
        if isinstance(criterion_name, str):
            criterion_name = [criterion_name]

        criterion_dict = {}
        for name in criterion_name:
            criterion_cls = criterions.registry[name]["cls"]
            config = criterions.registry[name]["default_config_dict"]
            criterion_dict[name] = criterion_cls(config)

        self.criterion = criterions.Criterion(criterion_dict, self.global_config.gpu_count)
        return self.criterion

    def build_hooks(self, hooks_name: List):
        self.hook_list = HookList()
        for name in hooks_name:
            hook = self.hook_factory(name)
            self.hook_list.add_hook(hook)

        # if log_hook exists, adjust log_hook as the last element in the container so that log_hook can
        # log all the output that other hooks modify.
        self.hook_list.move_hook_to_last(hooks.registry["log_hook"])

    def hook_factory(self, name):
        hook_cls = hooks.registry[name]
        if name == "log_hook":
            return hook_cls(self, self.global_config.log_interval, self.global_config.rank)
        elif name == "early_stop_hook":
            return hook_cls(self,
                            self.global_config.max_update,
                            self.global_config.stop_min_lr,
                            self.global_config.performance_decay_tolerance,
                            self.global_config.performance_indicator)
        elif name == "test_log_hook":
            return hook_cls(self.global_config.rank)
        elif name == "time_hook":
            return hook_cls()
        elif name == "tensorboard":
            if self.global_config.log_dir is None:
                self.global_config.log_dir = os.path.basename(self.global_config.ckpt_dir)
            return hook_cls(self, self.global_config.rank, self.global_config.log_dir, self.global_config.metrics_name)

    def build_metrics(self, metrics_name: List):
        self.metric_list = MetricList()
        for name in metrics_name:
            metric = self.metric_factory(name)
            self.metric_list.add_metric(name, metric)

    def metric_factory(self, name):
        metric_cls = metrics.registry[name]
        if name == "BLEU":
            return metric_cls(self.global_config.length_beam_size, self.tgt_dict, self.global_config.device)
        elif name == "length_accuracy":
            return metric_cls(self.global_config.length_beam_size, self.tgt_dict, self.global_config.device)

    def convert_batch_on_gpu(self, batch: Dict):
        for key, value in batch.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    batch[key][inner_key] = inner_value.to(device=self.global_config.device)
            else:
                batch[key] = value.to(device=self.global_config.device)

    def sync_sample_num(self, sample_num: List[Dict]):
        """

        :param sample_num: [{"ntokens": #tokens, "nsentences": #nsentences}]
        """
        sync_result = {}
        keys = sample_num[0].keys()

        # sum sample number in chunk
        for k in keys:
            temp = 0
            for el in sample_num:
                temp += el[k]
            sync_result[k] = temp

        # Synchronize sample number among different processes.
        for k in keys:
            dist.all_reduce(sync_result[k])

        return sync_result

    def eval_sync_loss(self, loss_list: List[Dict], loss_reduce_strategy: Dict, sync_sample_num: Dict) -> Dict:

        loss_dict = {}
        keys = loss_list[0].keys()
        # sum loss in chunk
        for k in keys:
            temp = 0
            for el in loss_list:
                temp += el[k]
            loss_dict[k] = temp

        # Synchronize loss among different processes.
        for k in keys:
            dist.all_reduce(loss_dict[k])

        # every criterion has its own average strategy, "ntokens" or "nsentences"
        for name, value in loss_dict.items():
            nsmples = sync_sample_num[loss_reduce_strategy[name]]
            loss_dict[name] = loss_dict[name] / nsmples

        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict

    def train_sync_loss(self, loss_list: List[Dict]) -> Dict:

        loss_dict = {}
        keys = loss_list[0].keys()
        count = len(loss_list)
        # sum loss in chunk
        for k in keys:
            temp = 0
            for el in loss_list:
                temp += el[k]
            loss_dict[k] = temp

        # Synchronize loss among different processes.
        for k in keys:
            dist.all_reduce(loss_dict[k])

        loss_dict["loss"] = sum(loss_dict.values())

        loss_dict = {key: value / (self.global_config.gpu_count * count) for key, value in loss_dict.items()}
        return loss_dict

    def train_epoch(self, checkpoint):
        train_epoch_iterator = self.train_data_handler.get_epoch_iterator()
        self.lr_scheduler.step_begin_epoch(self.train_data_handler.epoch_num)

        logging_outputs = {}
        self.hook_list.on_train_epoch_begin()

        for chunk in train_epoch_iterator:
            self.optimizer.zero_grad()
            loss = self.train_chunk(chunk, self.model)

            # loss.backward()
            self.optimizer.step()

            total_updates = self.train_data_handler.total_updates
            self.lr_scheduler.step_update(total_updates)
            self.model.decoder.y_ratio_scheduler.forward(total_updates)

            if self.validate:
                eval_output = self.eval(self.dev_data_handler, self.model, self.criterion)
                eval_output["epoch_end"] = self.train_data_handler.cur_iterator.epoch_end

                if self.save:
                    checkpoint.save(eval_output)

                if self.early_stop:
                    break

        self.hook_list.on_train_epoch_end(logging_outputs)

    def train_chunk(self, chunk: List, model):
        self.hook_list.on_train_chunk_begin()
        loss_list = []
        sample_num_list = []
        count = 0.
        for batch in chunk:
            loss_dict, sample_num_dict = self.train_step(batch, model)
            sample_num_list.append(sample_num_dict)
            loss_list.append(loss_dict)
            count += 1

        loss_dict: dict = self.train_sync_loss(loss_list)

        if hasattr(self._wrapped_model, "all_reduce_grads"):
            self._wrapped_model.all_reduce_grads() # noqa

        self.optimizer.multiply_grads(1 / count)

        self.hook_list.on_train_chunk_end(loss_dict)
        return loss_dict["loss"]

    def train_step(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model) -> Tuple[Dict, Dict]:
        self.hook_list.on_train_batch_begin()

        self.convert_batch_on_gpu(batch)

        model.train()
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
        #                                      profile_memory=True) as prof:
        # with autocast():
        model_outputs: dict = model(batch["net_input"]["src_tokens"],
                                    batch["net_input"]["prev_tgt_tokens"])
        loss, loss_dict = self.criterion.train(model_outputs, batch["target"], batch["tgt_lengths"],
                                               self.tgt_dict.padding_id, batch["nsentences"], batch["ntokens"])
        loss.backward()
        del loss
        # print(prof.table())
        # prof.export_chrome_trace('./test.json')


        sample_num_dict = {"nsentences": batch["nsentences"],
                           "ntokens": batch["ntokens"]}

        self.hook_list.on_train_batch_end()
        return loss_dict, sample_num_dict

    def eval(self, dev_data_handler, model, criterion):
        self.hook_list.on_eval_begin()
        self.metric_list.reset()

        dev_epoch_iterator = dev_data_handler.get_epoch_iterator()
        model.eval()

        loss_colletion, sample_num_colletion = [], []
        with torch.no_grad():
            for chunk in dev_epoch_iterator:
                loss_list, sample_num_list, hypo_list = self.eval_chunk(chunk, model, criterion)
                for batch, hypos in zip(chunk, hypo_list):
                    self.metric_list.update(batch["target"], hypos)

                loss_colletion.extend(loss_list)
                sample_num_colletion.extend(sample_num_list)

        sync_sample_num: dict = self.sync_sample_num(sample_num_colletion)
        loss_dict: dict = self.eval_sync_loss(loss_colletion, self.criterion.reduce_strategy, sync_sample_num)

        for name, metric in self.metric_list.items():
            loss_dict[name] = metric.result()

        self.hook_list.on_eval_end(loss_dict)
        return loss_dict

    def eval_chunk(self, chunk: List, model, criterion):
        self.hook_list.on_eval_chunk_begin()
        loss_list, sample_num_list, hypo_list = [], [], []

        for batch in chunk:
            loss_dict, sample_num_dict, hypo_tokens = self.eval_step(batch, model, criterion)
            loss_list.append(loss_dict)
            sample_num_list.append(sample_num_dict)
            hypo_list.append(hypo_tokens)

        self.hook_list.on_eval_chunk_end()
        return loss_list, sample_num_list, hypo_list

    def eval_step(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model, criterion):
        self.hook_list.on_eval_batch_begin()

        self.convert_batch_on_gpu(batch)
        model.eval()
        model_outputs: dict = model(batch["net_input"]["src_tokens"],
                                    batch["net_input"]["prev_tgt_tokens"])
        loss_dict: dict = self.criterion.eval(model_outputs, batch["target"], batch["tgt_lengths"],
                                               self.tgt_dict.padding_id)

        decoding_outputs: dict = model.generate(batch["net_input"]["src_tokens"])

        sample_num_dict = {"nsentences": batch["nsentences"],
                           "ntokens": batch["ntokens"]}

        hypo_tokens = decoding_outputs["hypo_tokens"]

        self.hook_list.on_eval_batch_end()
        return loss_dict, sample_num_dict, hypo_tokens

    def inference(self, test_data_handler, model, criterion):
        """
        we keep this inference function for extension
        """
        self.hook_list.on_inference_begin()

        test_epoch_iterator = test_data_handler.get_epoch_iterator()
        model.eval()

        loss_dict = {}
        with torch.no_grad():
            for chunk in test_epoch_iterator:
                hypo_list = self.inference_chunk(chunk, model, criterion)
                for batch, hypos in zip(chunk, hypo_list):
                    self.metric_list.update(batch["target"], hypos)

        for name, metric in self.metric_list.items():
            loss_dict[name] = metric.result()

        self.hook_list.on_inference_end(loss_dict)
        return loss_dict

    def inference_chunk(self, chunk: List, model, criterion):

        self.hook_list.on_inference_chunk_begin()

        hypo_list = []
        for batch in chunk:
            hypo_tokens = self.inference_step(batch, model, criterion)
            hypo_list.append(hypo_tokens)

        self.hook_list.on_inference_chunk_end()

        return hypo_list

    def inference_step(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model, criterion):
        self.hook_list.on_inference_batch_begin()

        self.convert_batch_on_gpu(batch)
        model.eval()
        decoding_outputs: dict = model.generate(batch["net_input"]["src_tokens"])

        hypo_tokens = decoding_outputs["hypo_tokens"]
        self.hook_list.on_inference_batch_end()

        return hypo_tokens

    def state_dict(self, train_iterator, model, optimizer, checkpoint) -> Dict:
        # basically, we need save the dataset iterator state, model state, criterion state, optimizer state,
        # lr scheduler state, maybe other config arguments. These are now in a mess, we should arrange them
        # in order.
        return {
            "train_iterator": train_iterator.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "checkpoint": checkpoint.state_dict()
        }

    def load_state_dict(self, state_dict: Dict, model, train_iterator=None, optimizer=None, checkpoint=None,
                        reset=False) -> None:
        model.load_state_dict(state_dict["model"], strict=False)
        if train_iterator and not reset:
            train_iterator.load_state_dict(state_dict["train_iterator"])
        if optimizer and not reset:
            optimizer.load_state_dict(state_dict["optimizer"])
        if checkpoint:
            checkpoint.load_state_dict(state_dict["checkpoint"])

    @property
    def validate(self) -> bool:
        res = (self.global_config.validate_update_interval != 0 and
               self.train_data_handler.total_updates % self.global_config.validate_update_interval == 0) \
              or self.train_data_handler.cur_iterator.epoch_end
        return res

    @property
    def save(self) -> bool:
        save_after_epoch_cond = (self.train_data_handler.epoch_num > self.global_config.save_after_epoch)

        save_interval_updates_cond = (self.global_config.save_interval_updates != 0 and
                 self.train_data_handler.rank_total_updates % self.global_config.save_interval_updates == 0)

        save_interval_cond = (self.global_config.save_interval != 0 and
                 self.train_data_handler.epoch_num % self.global_config.save_interval == 0)

        res = save_after_epoch_cond and (save_interval_updates_cond or save_interval_cond)
        return res

    @property
    def is_training_state(self) -> bool:
        # here we cab only control early stop condition on epoch number level
        res = self.train_data_handler.epoch_num < self.global_config.max_epoch \
              and (not self.early_stop)

        return res
