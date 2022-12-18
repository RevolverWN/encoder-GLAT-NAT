import argparse
import logging
import sys

import torch
import torch.distributed as dist

import criterions
import lr_schedulers
import models
import optimizers
import options
import tasks
from utils import Checkpoint, CheckpointFairseq

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    stream=sys.stdout)
# file_handler = logging.FileHandler()
logger = logging.getLogger("train")

MODULE_DICT = {"task": tasks,
               "model": models,
               "criterion": criterions,
               "lr_scheduler": lr_schedulers,
               "optimizer": optimizers}


class ArgsAdapter(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def add_specific_args(parser: argparse.ArgumentParser, known_args: argparse.Namespace):
    for name, module in MODULE_DICT.items():
        if not hasattr(module, "registry"):
            raise AttributeError("module {} has not registry attribute".format(module.__name__))
        if not hasattr(known_args, name):
            raise AttributeError("args has not {} attribute".format(name))

        # cls_name maybe str or tuple
        cls_name = getattr(known_args, name)
        if isinstance(cls_name, str):
            module.registry[cls_name]["cls"].add_args(parser)  # noqa
        elif isinstance(cls_name, (tuple, list)):
            for el in cls_name:
                module.registry[el]["cls"].add_args(parser)  # noqa


def main(proc_id: int, args: argparse.Namespace):
    # 1 setup device and distributed training
    if args.device == "cuda":
        args.device = "cuda:{}".format(proc_id)
        torch.cuda.set_device(args.device)

    args.rank = proc_id
    dist.init_process_group(backend=args.backend,
                            init_method=args.init_method,
                            world_size=args.gpu_count,
                            rank=args.rank)

    # 2 reset all modules configuration and build all modules
    for name, module in MODULE_DICT.items():
        cls_name = getattr(args, name)
        if isinstance(cls_name, str):
            module_default_config = module.registry[cls_name]["default_config_dict"]
            module_config = {args_name: getattr(args, args_name) for args_name in module_default_config.keys()}
            config = ArgsAdapter(**module_config)
            module.registry[cls_name]["default_config_dict"] = config
        elif isinstance(cls_name, (tuple, list)):
            for el in cls_name:
                module_default_config = module.registry[el]["default_config_dict"]
                module_config = {args_name: getattr(args, args_name) for args_name in module_default_config.keys()}
                config = ArgsAdapter(**module_config)
                module.registry[el]["default_config_dict"] = config

    logger.info(args)
    task_cls = tasks.registry[args.task]["cls"]
    config = tasks.registry[args.task]["default_config_dict"]
    task = task_cls(args, config)

    task.load_dataset(args.train_prefix)
    task.load_dataset(args.valid_prefix)
    task.load_dataset(args.test_prefix)

    train_data_handler = task.build_data_handler(args.train_prefix, args.train_chunk_size, args.max_tokens, args.max_sentences,
                                                 args.rank, args.gpu_count, args.device, args.buffer_size)
    dev_data_handler = task.build_data_handler(args.valid_prefix, args.dev_chunk_size, args.max_tokens, args.max_sentences,
                                               args.rank, args.gpu_count, args.device, args.buffer_size)
    test_data_handler = task.build_data_handler(args.test_prefix, args.test_chunk_size, args.max_tokens, args.max_sentences,
                                                args.rank, args.gpu_count, args.device, args.buffer_size)

    logger.info("{} samples in train dataset".format(len(train_data_handler.dataset)))
    logger.info("{} batches(updates) in every gpu train dataset over {} gpu".format(train_data_handler.rank_batches_num,
                                                                         train_data_handler.gpu_count))
    logger.info("{} samples in dev dataset".format(len(dev_data_handler.dataset)))
    logger.info("{} samples in test dataset".format(len(test_data_handler.dataset)))

    model = task.build_model(args.model)
    criterion = task.build_criterion(args.criterion)
    optimizer = task.build_optimizer(args.optimizer, model)
    lr_scheduler = task.build_lr_scheduler(args.lr_scheduler, optimizer)
    task.build_hooks(args.hooks_name)
    task.build_metrics(args.metrics_name)

    # 3 load state_dict and start training
    checkpoint = CheckpointFairseq(args.ckpt_dir, args, task, model, args.save_metric, optimizer, train_data_handler)
    checkpoint.load_checkpoint(args.ckpt_name)

    train_logging_outputs = {}
    task.hook_list.on_train_begin()
    while task.is_training_state:
        task.train_epoch(checkpoint)

    logger.info("start inference")
    task.inference(test_data_handler, model, criterion)
    task.hook_list.on_train_end(train_logging_outputs)


def args_postprocess(args: argparse.Namespace):
    args.max_tokens = int(args.max_tokens / args.gpu_count)


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(allow_abbrev=False)
    options.get_train_options(parser)
    known_args, _ = parser.parse_known_args()
    add_specific_args(parser, known_args)

    args = parser.parse_args()
    # args_postprocess(args)
    torch.multiprocessing.spawn(fn=main, args=(args,), nprocs=args.gpu_count)
