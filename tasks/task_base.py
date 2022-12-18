from typing import Tuple

import criterions
import lr_schedulers
import optimizers
from dataset.iterator import DataHandler
from dictionary import Dictionary
from criterions import registry


class TaskBase(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = {}

        self.train_data_handler = None
        self.dev_data_handler = None
        self.test_data_handler = None
        self.model = None
        self.criterion = {}  # store more criterion in a dict
        self.optimizer = None
        self.lr_scheduler = None
        self.hook_list = None
        self.metric_list = None

    def load_dataset(self, split):
        raise NotImplementedError

    def build_model(self, model_name):
        raise NotImplementedError

    def build_optimizer(self, optimizer_name, model):
        optimizer_cls = optimizers.registry[optimizer_name]["cls"]
        config = optimizers.registry[optimizer_name]["default_config_dict"]
        self.optimizer = optimizer_cls(model.parameters(), config)
        return self.optimizer

    def build_lr_scheduler(self, lr_scheduler_name, optimizer):
        lr_scheduler_cls = lr_schedulers.registry[lr_scheduler_name]["cls"]
        config = lr_schedulers.registry[lr_scheduler_name]["default_config_dict"]
        self.lr_scheduler = lr_scheduler_cls(optimizer, config)
        return self.lr_scheduler

    def state_dict(self, train_iterator, model, optimizer, checkpoint):
        # basically, we need save the dataset iterator state, model state, optimizer state, while criterion state and
        # lr scheduler state, they are no inner states vary in training stage, so they can initialize from
        pass

    def load_state_dict(self, state_dict, train_iterator, model, optimizer, checkpoint):
        pass
