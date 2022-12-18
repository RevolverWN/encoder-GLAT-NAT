import torch


class LrSchedulerBase(object):

    def __init__(self):
        pass

    def step_update(self, num_updates):
        pass

    def step_begin_epoch(self, epoch):
        pass

    def state_dict(self):
        pass

    def reset(self):
        pass