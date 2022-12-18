import copy
import itertools
import logging
import math
import queue
import time
from threading import Thread
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import deprecated

logger = logging.getLogger(__name__)

_sentinel = object()


class ChunkIterWrapper(object):
    def __init__(self, iterator, chunk_size: int, total_chunk: int, count: int = 0):
        self.iterator = iterator
        self.chunk_size = chunk_size
        self.total_chunk = total_chunk
        self.chunk_count = count
        self.epoch_end = False

        # if self.chunk_count > 0:
        #     raise ValueError("chunk_count must be 0, bypass method has been deprecated!")
            # self.bypass(count)

    # bypass method has been deprecated, this method iterates self.iterator up to chunk_count which will be
    # exceedingly time-consuming if __getitem__ or collate_fn method in Dataset class time-consuming, so we slice
    # _index_sampler in Dataloader to truncate data flow.
    @deprecated
    def bypass(self, chunk_count):
        for i in range(chunk_count):
            chunk_list = []
            try:
                for j in range(self.chunk_size):
                    chunk_list.append(next(self.iterator))
            except StopIteration:
                pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.chunk_count < self.total_chunk:
            if self.chunk_count == self.total_chunk - 1:
                self.epoch_end = True

            chunk_list = []
            try:
                for i in range(self.chunk_size):
                    chunk_list.append(next(self.iterator))
            except StopIteration:
                pass

            self.chunk_count += 1
            return chunk_list
        else:
            raise StopIteration


class DataHandler(object):

    def __init__(self,
                 dataset,
                 chunk_size: int,
                 rank: int,
                 device,
                 gpu_count: int,
                 max_tokens=None,
                 max_sentences=None,
                 buffer_size=0,
                 train_flag=False):

        if bool(max_sentences) == bool(max_tokens):
            raise ValueError("only one of max_tokens and max_sentences can be assigned")

        self.dataset = dataset
        self.chunk_size = chunk_size
        self.device = device
        self.gpu_count = gpu_count
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.train_flag = train_flag

        self.buffer_size = buffer_size
        self.indices_batches: List[List[int]] = self.dataset.batch_sampler(max_tokens=max_tokens,
                                                                           max_sentences=max_sentences)
        self.batches_num: int = len(self.indices_batches)

        if gpu_count > 1 and (self.batches_num % gpu_count) != 0:
            self.rearrange_indices_batches()

        self.batches_by_shards: List[List[List[int]]] = [self.indices_batches[start: self.batches_num: gpu_count]
                                                         for start in range(gpu_count)]

        assert all(len(self.batches_by_shards[0]) == len(mini_batch) for mini_batch in self.batches_by_shards), \
            "minibatches are not the same size"

        self.rank_batches: List[List[int]] = self.batches_by_shards[rank]
        self.rank_batches_num = len(self.rank_batches)

        self.epoch_num = 0

        self.rank_total_chunks = math.ceil(self.rank_batches_num / chunk_size)
        self.cur_iterator = ChunkIterWrapper(self._get_iter(), self.chunk_size, self.rank_total_chunks, count=0)

    def rearrange_indices_batches(self):

        remainder_num = self.batches_num % self.gpu_count
        remainder_batches = self.indices_batches[-remainder_num:]
        if self.max_sentences:
            total_sentences = sum(len(indices_batch) for indices_batch in remainder_batches)
            sent_num_per_gpu = math.ceil(total_sentences / self.gpu_count)
            flatten_indices_batches = list(itertools.chain.from_iterable(remainder_batches))
            new_remainder = []
            for i in range(self.gpu_count):
                new_remainder.append(flatten_indices_batches[i * sent_num_per_gpu: (i + 1) * sent_num_per_gpu])
            self.indices_batches = self.indices_batches[: -remainder_num] + new_remainder
            self.batches_num: int = len(self.indices_batches)

        if self.max_tokens:
            while True:
                last_n_batch = np.concatenate(remainder_batches)
                if len(last_n_batch) >= self.gpu_count:
                    new_remainder = np.array_split(last_n_batch, self.gpu_count)
                    self.indices_batches = self.indices_batches[: -remainder_num] + new_remainder
                    self.batches_num: int = len(self.indices_batches)
                    break

                remainder_num = remainder_num + self.gpu_count
                remainder_batches = self.indices_batches[-remainder_num:]

    @property
    def rank_total_updates(self):
        res = self.rank_total_chunks * (self.epoch_num - 1) + self.cur_iterator.chunk_count
        return res

    @property
    def total_updates(self):
        return self.rank_total_updates

    @property
    def updates_in_epoch(self):
        return self.cur_iterator.chunk_count

    def state_dict(self) -> Dict:
        return {'cur_epoch': self.epoch_num,
                'updates_in_epoch': self.cur_iterator.chunk_count,
                'rank_total_updates': self.rank_total_updates}

    def load_state_dict(self, state_dict: Dict):
        count = state_dict['updates_in_epoch']

        self.epoch_num = state_dict['cur_epoch']
        # initialize self.cur_iterator with self.epoch_num = 0, now self.epoch_num has changed, so we must reset
        # self.cur_iterator
        self.cur_iterator = ChunkIterWrapper(self._get_iter(), self.chunk_size, self.rank_total_chunks, count)

        if count > 0:
            # self.cur_iterator.bypass(count)
            bypass_batch_num = self.chunk_size * count

            if isinstance(self.cur_iterator.iterator, BufferedIterator):
                self.cur_iterator.iterator._iterable._index_sampler = self.cur_iterator.iterator._iterable._index_sampler[bypass_batch_num:]
            else:
                self.cur_iterator.iterator._index_sampler = self.cur_iterator.iterator._index_sampler[
                                                                      bypass_batch_num:]

    def get_epoch_iterator(self):
        """
        There are three conditions:
        the first is BatchStateIterator initialized associated with StateEpochIterator, so get_epoch_iterator should
        make self.cur_epoch plus one, then return self.cur_batch_iterator
        the second is self.cur_batch_iterator is restored from state_dict and the offset is less than total number,
        it means that the iteration is still in self.cur_epoch, just return self.cur_batch_iterator
        the third is self.cur_batch_iterator has been exhausted either from outer loop in run time or restored from
        state_dict while the offset happens equal to total number. Thus we should make self.cur_epoch plus one and
        initialize a new BatchStateIterator object assigned to self.cur_batch_iterator, then return
        self.cur_batch_iterator
        :return:
        """
        if self.cur_iterator.chunk_count == 0:
            self.epoch_num += 1

        elif self.cur_iterator.chunk_count == self.rank_total_chunks:
            self.epoch_num += 1
            self.cur_iterator = ChunkIterWrapper(self._get_iter(), self.chunk_size, self.rank_total_chunks, count=0)

        return self.cur_iterator

    def _get_iter(self):
        batches = copy.deepcopy(self.rank_batches)
        np.random.seed(self.epoch_num)
        if self.train_flag:
            np.random.shuffle(batches)

        iterator = iter(DataLoader(dataset=self.dataset,
                                   batch_sampler=batches,  # noqa
                                   collate_fn=self.dataset.collate_fn,
                                   pin_memory=True))
        if self.buffer_size > 0:
            iterator = BufferedIterator(self.buffer_size, iterator) # noqa
        return iterator


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len, cuda_device):
        Thread.__init__(self)

        self._queue = queue
        self._source = source
        self._max_len = max_len
        self.count = 0
        self.cuda_device = cuda_device

    def run(self):
        # set_device to avoid creation of GPU0 context when using pin_memory
        if self.cuda_device is not None:
            torch.cuda.set_device(self.cuda_device)

        try:
            for item in self._source:
                self._queue.put(item)

                # Stop if we reached the maximum length
                self.count += 1
                if self._max_len is not None and self.count >= self._max_len:
                    break

            # Signal the consumer we are done.
            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)


class BufferedIterator(object):
    def __init__(self, size, iterable):
        self._queue = queue.Queue(size)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

        self.total = len(iterable)

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(
            self._queue,
            self._iterable,
            self.total,
            torch.cuda.current_device() if torch.cuda.is_available() else None
        )
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def take(self, n):
        self.total = min(self.total, n)
        # Propagate this change to the underlying iterator
        if hasattr(self._iterable, "take"):
            self._iterable.take(n)
        return self

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()

        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < min(2, max(1, self._queue.maxsize // 2)):
            if time.time() - self.start_time > 5 * 60:
                if (
                        self.warning_time is None
                        or time.time() - self.warning_time > 15 * 60
                ):
                    logger.debug(
                        "Data loading buffer is empty or nearly empty. This may "
                        "indicate a data loading bottleneck, and increasing the "
                        "number of workers (--num-workers) may help."
                    )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get(True)
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration()
        return item


if __name__ == '__main__':
    pass
