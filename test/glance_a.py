import inspect
import json
import math
import operator
import os
import pickle
import sys
import argparse
import itertools
from dataclasses import dataclass, field
from pprint import pprint
from typing import TypeVar, Callable, Any

from hydra.experimental import initialize, compose

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# training time, progress bar, outputs

# class ArgsWrapper(object):
#     def __init__(self, attr_dict:dict):
#         self.attr_dict = attr_dict
#
#     def __getattribute__(self, key):
#         return self.attr_dict[key]



# a = {"a": 1, "b": 2}
#
# args = ArgsWrapper(a)
#
# print(args.a)


# class Foo:
#     def __init__(self):
#         self.name = "wang"
#
#     def __setstate__(self, state):
#         self.name = state
#
#     def __getstate__(self):
#         return self.name
#
# foo = Foo()
#
# def main(i, args):
#     args.rank = i
#     dist.init_process_group(backend="gloo", rank=args.rank, world_size=args.nprocs, init_method="tcp://127.0.0.1:12345")
#
#     a = torch.tensor([1, 2])
#     dist.all_reduce(a)
#     print(a)
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--nprocs", default=1, type=int)
#     parser.add_argument("--rank", default=0, type=int)
#     args = parser.parse_args()
#     torch.multiprocessing.spawn(fn=main, nprocs=args.nprocs, args=(args,))

# a = [[1, 2], [2, 4], [5, 4], [3, 6], [8, 9]]
# shares = 3
# a_length = len(a)
#
# interval = math.ceil(a_length / 3)
#
# groups = [a[start: a_length: interval+1] for start in range(shares)]
# groups = itertools.zip_longest(groups, fillvalue=3)
# print(list(groups))
#
# itr = map(
#             operator.itemgetter(1),
#             itertools.zip_longest(
#                 range(a_length),
#                 itertools.islice(a, 1, a_length, shares),
#                 fillvalue=[],
#             ),
#         )
# print(list(itr))


# def collate_fn(batch):
#     if batch:
#         return batch
#     else:
#         return []
#
# class Data(Dataset):
#     def __init__(self):
#         self.data = [1, 2, 4, 6]
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return len(self.data)
#
# data = Data()
#
# data_loader = DataLoader(dataset=data, batch_sampler=[[1, 2], [0, 1], [], []], collate_fn=collate_fn)
#
# for i in data_loader:
#     print(i)

# 1 for 循环index error问题
# 2 轮空 优化器优化

# class Data(object):
#     def __init__(self):
#         self.data = [1, 2, 5, 6]
#         self.length = 6
#         self.count = 0
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.count < self

# class Data:
#     def __init__(self, it):
#         self.it = it
#         self.cur_data = self.it.__next__()
#         self.cur_shape = self.cur_data[0].shape()
#
#     def cache(self):
#         self.it.__next__()
#         self.cur_data = self.it.__next__()
#         self.cur_shape = self.cur_data[0].shape()
#
#     def __next__(self):
#         previous_data = self.cur_data
#         self.cache()
#         return previous_data

# class Model(nn.Module):
#
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(2, 2, bias=False)
#
#     def forward(self, x):
#         return self.fc1(x)
# #
# #
# n_proc = 2
# batch_size = 3
# features = 2
#
# data = torch.randn(n_proc, batch_size, features)
# label = torch.randint(0, 2, size=(n_proc, batch_size))
# model = Model()
# device = torch.device("cuda")
# model.to(device=device)
# data.to()
# print('hh')
# criterion = nn.CrossEntropyLoss()
#
#
# def main(i):
#     dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:15000", world_size=n_proc, rank=i)
#
#     logits = model(data[i])
#     loss = criterion(logits, label[i])
#     loss.backward()
#     for name, params in model.named_parameters():
#         print(name, params.grad)
#         dist.all_reduce(params.grad)
#         print(name, params.grad)
#
#
# if __name__ == '__main__':
#     torch.multiprocessing.spawn(fn=main, nprocs=2)
# import torch
# from torch import nn
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(1, 2)
#         self.fc2 = nn.Linear(2, 3)
#dataclass
#     def forward(self):
#         pass
#
#
# model = Model()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# print(type(optimizer.param_groups))

# wmt16 训练参数和预处理
#
# parser = argparse.ArgumentParser()
# parser.add_argument_group()

# @dataclass(frozen=True, order=True)
# class Comment(object):
#     id: int = field()
#     text: str
#
#
# comment = Comment(1, "hello")
# pprint(inspect.getmembers(Comment, inspect.isfunction))
# print(comment)

# initialize("")
#
# cfg = compose("config.yaml", overrides=["db=mysql", "db.user=me"])
# print(OmegaConf.to_yaml(cfg))

# token数量决定batch多少，也就是决定step多少，而step多少决定了学习率更新的快慢


# 目前程序跑通了，现在想办法把bleu值调上去，还有文献里计算速度的方法
# 影响bleu值的因素有以下几个：
# 1 对解析出来的字符串的后处理。
# 2 不同模型的影响
# 3 参数调节对模型的影响
# 目前对这两个因素无法判断是哪个影响更大一些，但是都是要处理的


# 1 不应该忽略没遮盖的那些词，应该鼓励他们，这是一个MLM里的普遍问题。
# 2 比较不同长度策略对预测的影响，这相当于是一个新问题，一定要说明这个问题的意义在哪里
#  文献里从来没有在开发集上验证长度的准确率对于bleu的影响，可以把它单独拿出来作为一个问题研究。
# 3 概率大的不一定就是准确的预测，概率小的也不一定就不可靠，验证是否存在这个问题。
# 4 结合自回归和非自回归
# 5 损失截断
# 6 共享编码器
# 7 提前加损失函数
# 8 mention flag

# 2021.11.14
# 1 loss-guide，目前采样方法受阻，导致性能并不好
    # 是否存在采样不均匀的问题，如何验证
# 2 损失截断讨论:设置比例启动截断，比例为损失值
    # 对蒸馏数据和非蒸馏数据做loss图的对比，在波动大的时候加入损失截断
# 3 结合回归和非自回归的影响
    # 代码问题
# 4 结合n-gram损失：设置n-gram的reward
# 5 让复制的embedding和真正的目标embedding KL距离越小损失
    #
# 6 共享编码器

# 2021.11.17
#
import torch
torch.optim.lr_scheduler.LambdaLR


