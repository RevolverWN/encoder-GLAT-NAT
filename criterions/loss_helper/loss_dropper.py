import numpy as np
import torch.nn as nn


class LossDropper(nn.Module):
    def __init__(
            self,
            dropc=0.4,
            min_count=10000,
            buffer_size=10000,
            verbose=True
    ):
        super().__init__()
        self.keepc = 1. - dropc
        self.count = 0
        self.min_count = min_count

        self.buffer_size = buffer_size
        self.accumulative_computed = 0
        self.percentile_val = 100000000.
        self.buffer_cur_idx = 0

        self.verbose = verbose

        self.buffer_zone = np.zeros(self.buffer_size, dtype=np.float32)

    def forward(self, loss):
        """
        四条触发返回：
        1 loss是none
        2 队列不满
        3 队列满了但是没达到最小要求数量
        4 队列满了，返回

        重新计算分位值的触发条件，缓冲区满了
        :param loss:
        :return:
        """
        if loss is None:
            return loss

        self.accumulative_computed += loss.numel()
        self.count += loss.numel()
        # 加入缓冲区
        if self.count < len(self.buffer_zone):
            self.buffer_zone[self.count - loss.numel():self.count] = loss.detach().cpu().numpy().flatten()
            self.buffer_cur_idx += loss.numel()
            return (loss < np.inf).type(loss.dtype)
        # 缓冲区满了，接着按照顺序放loss值，到缓冲区结尾后再从头放
        else:
            for idx, item in enumerate(loss):
                self.buffer_zone[self.buffer_cur_idx] = item
                self.buffer_cur_idx += 1
                if self.buffer_cur_idx >= len(self.buffer_zone):
                    self.buffer_cur_idx = 0

        # 队列满了，小于最小值
        if self.count < self.min_count:
            return (loss < np.inf).type(loss.dtype)
        #
        # 大于最小值
        if self.accumulative_computed > self.buffer_size:
            self.percentile_val = np.percentile(self.buffer_zone, self.keepc * 100)
            if self.verbose:
                print('Using cutoff', self.percentile_val)
            self.accumulative_computed = 0

        mask = (loss < self.percentile_val).type(loss.dtype)
        return mask
