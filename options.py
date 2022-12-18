import argparse
import math
import random

import torch

import tasks
import models
import criterions
import lr_schedulers
import optimizers
import hooks
import metrics


def get_preprocess_options(parser: argparse.ArgumentParser) -> None:
    # dictionary argument setup
    parser.add_argument("--src-lang", type=str, help="")
    parser.add_argument("--tgt-lang", type=str, help="")
    parser.add_argument("--joined-dictionary", action='store_true', help="learn joined dictionary")
    parser.add_argument("--corpus", type=str, help="corpus path")
    parser.add_argument("--src-min-frequency", type=int, default=0, help="")
    parser.add_argument("--tgt-min-frequency", type=int, default=0, help="")
    parser.add_argument("--src-dict", type=str, help="source dictionary path")
    parser.add_argument("--tgt-dict", type=str, help="target dictionary path")

    parser.add_argument("--destdir", type=str, default="data-bin", help="")
    parser.add_argument("--train-prefix", type=str, default="train",
                        help="default format is train.lang, eg train.en")
    parser.add_argument("--valid-prefix", type=str, default="valid", help="valid prefix")
    parser.add_argument("--test-prefix", type=str, default="test", help="test prefix")


def get_train_options(parser: argparse.ArgumentParser) -> None:
    get_common_options(parser)

    parser.add_argument("--optimizer", type=str, default="adam", choices=optimizers.registry.keys(),
                        help="pick one optimizer from the optimizer registry")
    parser.add_argument("--lr-scheduler", type=str, default="inverse_square_root",
                        choices=lr_schedulers.registry.keys(),
                        help="pick one lr scheduler from the scheduler registry")
    parser.add_argument("--train-chunk-size", type=int, default=1,
                        help="every chunk size batches update the parameters")
    parser.add_argument("--dev-chunk-size", type=int, default=1,
                        help="every chunk size batches in dev dataset")
    parser.add_argument("--test-chunk-size", type=int, default=1,
                        help="every chunk size batches in test dataset")

    parser.add_argument("--hooks-name", type=str, nargs='+', default=("log_hook", "early_stop_hook", "tensorboard",
                                                                      "test_log_hook", "time_hook"),
                        choices=hooks.registry.keys(),
                        help="pick hooks from hooks package registry")

    # stop conditions
    parser.add_argument("--stop-min-lr", type=float, default=-1.0, help="minimum learning rate")
    parser.add_argument("--max-epoch", type=int, default=math.inf, help="max epoch to stop")
    parser.add_argument("--max-update", type=int, default=math.inf, help="force stop training at specified update")
    parser.add_argument("--performance-decay-tolerance", type=int, default=math.inf, help="early stop training if "
                                                                                          "valid performance "
                                                                                          "doesn't improve for N consecutive validation"
                                                                                          " runs")
    parser.add_argument("--performance-indicator", type=str, default="label_smoothing_loss", choices=["loss", "BLEU"],
                        help="")

    # log, validate and save setup
    parser.add_argument("--validate-epoch-interval", type=int, default=1, help="validate every N epochs")
    parser.add_argument("--validate-update-interval", type=int, default=0, help="validate every N updates")

    parser.add_argument("--test-epoch-interval", type=int, default=1, help="test every N epochs")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="every N updates to log out the results in training phase")
    parser.add_argument("--reset", action="store_true", help="reset train dataset iterator and optimizer")

    # checkpoint setup
    parser.add_argument("--save-metric", type=str, default="loss", choices=["loss", "BLEU"],
                        help="metric to use for saving 'best' checkpoints")
    parser.add_argument("--save-after-epoch", type=int, default=0, help="save after N epoch")
    parser.add_argument("--save-interval", type=int, default=1, help="save a checkpoint every N epochs")
    # parser.add_argument("--save-period", type=int, default=0, help="every N updates to save the state dict")
    parser.add_argument("--save-interval-updates", type=int, default=0, help="save a checkpoint (and validate) every N updates")
    parser.add_argument("--keep-interval-updates", type=int, default=-1,
                        help="keep the last N checkpoints saved with --save-interval-updates")
    parser.add_argument("--keep-last-epochs", type=int, default=-1,
                        help="keep last N epoch checkpoints")
    parser.add_argument("--keep-best-checkpoints", type=int, default=-1,
                        help="keep best N checkpoints based on scores")
    parser.add_argument("--no-epoch-checkpoints", action="store_true",
                        help="only store last and best checkpoints")
    parser.add_argument("--no-last-checkpoints", action="store_true",
                        help="don't store last checkpoints")
    parser.add_argument("--maximize-best-checkpoint-metric", action="store_true",
                        help="select the largest metric value for saving 'best' checkpoints")
    parser.add_argument("--keep-interval-updates-pattern", type=int, default=-1,
                        help="when used with --keep-interval-updates, skips deleting "
                    "any checkpoints with update X where X %% keep_interval_updates_pattern == 0")



def get_generation_options(parser: argparse.ArgumentParser) -> None:
    get_common_options(parser)
    parser.add_argument("--hooks-name", type=str, nargs='+', default=("test_log_hook",),
                        choices=hooks.registry.keys(),
                        help="pick hooks from hooks package registry")
    parser.add_argument("--test-chunk-size", type=int, default=1,
                        help="every chunk size batches update the parameters")


def get_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data_dir", type=str, help="data directory must include train, dev, test, dictionary files"
                                                   "and they must have the uniform filename format.")
    parser.add_argument("ckpt_dir", type=str, help="checkpoint save path")
    parser.add_argument("--ckpt_name", default="checkpoint_last.pt", help="checkpoint filename")
    parser.add_argument("--log_dir", default=None, help="tensorboard data directory")

    parser.add_argument("--train-prefix", type=str, default="train",
                        help="default format is train.lang, eg train.en")
    parser.add_argument("--valid-prefix", type=str, default="valid", help="valid prefix")
    parser.add_argument("--test-prefix", type=str, default="test", help="test prefix")

    parser.add_argument("--task", type=str, default="translation", choices=tasks.registry.keys(),
                        help="pick one task from task registry")
    parser.add_argument("--model", type=str, default="nat_transformer", choices=models.registry.keys(),
                        help="pick one model from model registry")
    parser.add_argument("--criterion", type=str, nargs="+", default="label_smoothed_cross_entropy",
                        choices=criterions.registry.keys(),
                        help="pick one criterion from criterion registry")

    parser.add_argument("--metrics-name", type=str, nargs='+', default=("BLEU", "length_accuracy"),
                        choices=metrics.registry.keys(),
                        help="pick hooks from hooks package registry")

    # optimize setup
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="training device")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max tokens to batch the sequences")
    parser.add_argument("--max-sentences", type=int, default=None, help="max sentences to batch the sequences")
    parser.add_argument("--clip-norm", type=float, default=0.0, help="clip threshold of gradients")
    parser.add_argument("--buffer_size", type=int, default=10, help="data buffer size")

    # distributed setup
    parser.add_argument("--gpu-count", type=int, default=torch.cuda.device_count(),
                        help="distributed training gpu conut")
    parser.add_argument("--backend", type=str, default="nccl", choices=("gloo", "mpi", "nccl"),
                        help="backend communication protocol")
    parser.add_argument("--rank", type=int, default=0, help="master host rank")
    parser.add_argument("--init-method", type=str, default="tcp://127.0.0.1:{}".format(random.randint(10000, 20000)),
                        help="distributed training initial method")

