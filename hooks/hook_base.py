import logging
from typing import Dict, List
import time
import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from hooks import register_hook
from utils import convert_logging_out

logger = logging.getLogger(__name__)


class HookList(object):
    """
    we can design custom behavior on every stage of train, dev, test
    """

    def __init__(self):
        self.hook_list = []

    def move_hook_to_last(self, hook_cls):
        hook_index = None
        for i, hook in enumerate(self.hook_list):
            if isinstance(hook, hook_cls):
                hook_index = i
                break

        if hook_index is not None:
            self.hook_list.append(self.hook_list.pop(hook_index))

    def add_hook(self, hook):
        self.hook_list.append(hook)

    def on_train_begin(self):
        for hook in self.hook_list:
            hook.on_train_begin()

    def on_train_end(self, logging_outputs):
        for hook in self.hook_list:
            hook.on_train_end(logging_outputs)

    def on_train_epoch_begin(self):
        for hook in self.hook_list:
            hook.on_train_epoch_begin()

    def on_train_epoch_end(self, logging_outputs):
        for hook in self.hook_list:
            hook.on_train_epoch_end(logging_outputs)

    def on_train_chunk_begin(self):
        for hook in self.hook_list:
            hook.on_train_chunk_begin()

    def on_train_chunk_end(self, logging_outputs):
        for hook in self.hook_list:
            hook.on_train_chunk_end(logging_outputs)

    def on_train_batch_begin(self):
        for hook in self.hook_list:
            hook.on_train_batch_begin()

    def on_train_batch_end(self, logging_outputs=None):
        for hook in self.hook_list:
            hook.on_train_batch_end(logging_outputs)

    def on_eval_begin(self):
        for hook in self.hook_list:
            hook.on_eval_begin()

    def on_eval_end(self, logging_outputs):
        for hook in self.hook_list:
            hook.on_eval_end(logging_outputs)

    def on_eval_chunk_begin(self):
        for hook in self.hook_list:
            hook.on_eval_chunk_begin()

    def on_eval_chunk_end(self, *args, **kwargs):
        for hook in self.hook_list:
            hook.on_eval_chunk_end()

    def on_eval_batch_begin(self):
        for hook in self.hook_list:
            hook.on_eval_batch_begin()

    def on_eval_batch_end(self):
        for hook in self.hook_list:
            hook.on_eval_batch_end()

    def on_inference_begin(self):
        for hook in self.hook_list:
            hook.on_inference_begin()

    def on_inference_end(self, logging_outputs):
        for hook in self.hook_list:
            hook.on_inference_end(logging_outputs)

    def on_inference_chunk_begin(self):
        for hook in self.hook_list:
            hook.on_inference_chunk_begin()

    def on_inference_chunk_end(self, *args, **kwargs):
        for hook in self.hook_list:
            hook.on_inference_chunk_end()

    def on_inference_batch_begin(self):
        for hook in self.hook_list:
            hook.on_inference_batch_begin()

    def on_inference_batch_end(self):
        for hook in self.hook_list:
            hook.on_inference_batch_end()


class HookBase(object):
    """
    we design this hook base class for every data flow pipeline in train, eval, test by
    AOP(Aspect Oriented Programming) design pattern
    """

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass

    def on_train_epoch_begin(self, *args, **kwargs):
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_train_chunk_begin(self):
        pass

    def on_train_chunk_end(self, *args, **kwargs):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def on_eval_begin(self, *args, **kwargs):
        pass

    def on_eval_end(self, *args, **kwargs):
        pass

    def on_eval_chunk_begin(self):
        pass

    def on_eval_chunk_end(self, *args, **kwargs):
        pass

    def on_eval_batch_begin(self):
        pass

    def on_eval_batch_end(self):
        pass

    def on_inference_begin(self):
        pass

    def on_inference_end(self, *args, **kwargs):
        pass

    def on_inference_chunk_begin(self):
        pass

    def on_inference_chunk_end(self, *args, **kwargs):
        pass

    def on_inference_batch_begin(self):
        pass

    def on_inference_batch_end(self):
        pass


@register_hook("log_hook")
class LogHook(HookBase):

    def __init__(self, task, log_interval: int, rank: int):
        self.task = task
        self.rank = rank
        self.log_interval = log_interval

    def on_train_chunk_end(self, logging_outputs: Dict) -> None:
        total_updates = self.task.train_data_handler.total_updates
        logging_outputs["lr"] = self.task.optimizer.param_groups[0]['lr']
        if total_updates % self.log_interval == 0:
            if self.rank == 0:
                logger.info("epoch:{} | total updates:{} | {}".format(
                    self.task.train_data_handler.epoch_num,
                    total_updates,
                    convert_logging_out(logging_outputs)))

    def on_train_end(self, logging_outputs: Dict) -> None:
        if self.rank == 0:
            logger.info(convert_logging_out(logging_outputs))

    def on_train_epoch_end(self, logging_outputs: Dict) -> None:
        if self.rank == 0:
            logger.info(convert_logging_out(logging_outputs))

    def on_eval_end(self, logging_outputs) -> None:
        if self.rank == 0:
            logger.info(convert_logging_out(logging_outputs))

    def on_inference_end(self, logging_outputs: Dict) -> None:
        if self.rank == 0:
            logger.info(convert_logging_out(logging_outputs))


@register_hook("early_stop_hook")
class EarlyStopHook(HookBase):
    def __init__(self, task, max_updates, stop_min_lr, max_count, performance_indicator, tolerance=0.,
                 counter=0, min_value=np.inf):
        self.task = task
        self.counter = counter
        self.min_value = min_value
        self.best_epoch = None
        self.max_count = max_count
        self.tolerance = tolerance
        self.performance_indicator = performance_indicator

        self.max_updates = max_updates
        self.stop_min_lr = stop_min_lr

    def on_train_chunk_end(self, logging_outputs: Dict) -> None:
        res = self.task.optimizer.param_groups[0]['lr'] <= self.stop_min_lr \
              or self.task.train_data_handler.total_updates >= self.max_updates

        if res:
            self.task.early_stop = True

    def on_eval_end(self, logging_outputs) -> None:
        metric_value = logging_outputs[self.performance_indicator]
        metric_value = -metric_value if self.performance_indicator == "BLEU" else metric_value

        if self.min_value - metric_value > self.tolerance:
            self.min_value = metric_value
            self.best_epoch = self.task.train_data_handler.epoch_num
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.max_count:
                self.task.early_stop = True

        logging_outputs["best " + self.performance_indicator] = abs(self.min_value)
        logging_outputs["best performance epoch"] = self.best_epoch
        logging_outputs["tolerance decay count"] = self.counter


@register_hook("test_log_hook")
class TestLogHook(HookBase):
    def __init__(self, rank):
        self.rank = rank

    def on_inference_end(self, logging_outputs: Dict) -> None:
        if self.rank == 0:
            logger.info(convert_logging_out(logging_outputs))


@register_hook("time_hook")
class Timer(HookBase):
    def on_train_begin(self):
        self.train_time_start = time.time()

    def on_train_end(self, logging_outputs: Dict):
        self.train_time_end = time.time()
        total_time = self.train_time_end - self.train_time_start
        logging_outputs["total train time"] = str(datetime.timedelta(seconds=total_time))

    def on_train_epoch_begin(self):
        self.train_epoch_time_start = time.time()

    def on_train_epoch_end(self, logging_outputs: Dict):
        self.train_epoch_time_end = time.time()
        total_time = self.train_epoch_time_end - self.train_epoch_time_start
        logging_outputs["train epoch time"] = str(datetime.timedelta(seconds=total_time))

    def on_eval_begin(self):
        self.eval_epoch_time_start = time.time()

    def on_eval_end(self, logging_outputs: Dict):
        self.eval_epoch_time_end = time.time()
        total_time = self.eval_epoch_time_end - self.eval_epoch_time_start
        logging_outputs["eval time"] = str(datetime.timedelta(seconds=total_time))

    def on_inference_begin(self):
        self.inference_epoch_time_start = time.time()

    def on_inference_end(self, logging_outputs: Dict):
        self.inference_epoch_time_end = time.time()
        total_time = self.inference_epoch_time_end - self.inference_epoch_time_start
        logging_outputs["inference time"] = str(datetime.timedelta(seconds=total_time))


@register_hook("tensorboard")
class Tensorboard(HookBase):
    def __init__(self, task, rank, log_dir, metrics_name):
        self.task = task
        self.rank = rank
        self.metrics_name = metrics_name
        if rank == 0:
            self.train_writer = SummaryWriter(log_dir=f'logs/{log_dir}_train')
            self.valid_writer = SummaryWriter(log_dir=f'logs/{log_dir}_valid')

    def on_train_chunk_end(self, logging_outputs: Dict):
        total_updates = self.task.train_data_handler.total_updates
        if self.rank == 0:

            self.train_writer.add_scalar('lr', self.task.optimizer.param_groups[0]['lr'], total_updates)
            for name, loss in logging_outputs.items():
                self.train_writer.add_scalar(name, loss, total_updates)

    def on_eval_end(self, logging_outputs: Dict):
        total_updates = self.task.train_data_handler.total_updates
        if self.rank == 0:
            self.valid_writer.add_scalar('loss', logging_outputs["loss"], total_updates)
            for metric in self.metrics_name:
                self.valid_writer.add_scalar(metric, logging_outputs[metric], total_updates)



