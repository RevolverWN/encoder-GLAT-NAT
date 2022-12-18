import itertools
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from fairseq.data.data_utils import batch_by_size
from fairseq.data.data_utils_fast import (
    batch_by_size_fn,
    batch_by_size_vec,
    batch_fixed_shapes_fast,
)

from dictionary import Dictionary

logger = logging.getLogger(__name__)


def single_dataset_batch(sent_lengths, max_tokens, max_sentences, num_tokens_fn):
    if bool(max_sentences) == bool(max_tokens):
        raise ValueError("only one of max_tokens and max_sentences can be assigned")

    ordered_indices = np.argsort(sent_lengths)
    batches = batch_by_size(indices=ordered_indices,
                            num_tokens_fn=num_tokens_fn,
                            max_sentences=max_sentences,
                            max_tokens=max_tokens,
                            required_batch_size_multiple=8)
    return batches


class Seq2seqDataset(Dataset):
    def __init__(self, data_path: str, dictionary: Dictionary):
        self.data_path = data_path
        self.dictionary = dictionary

    def __getitem__(self, index: int) -> np.array:
        pass

    def num_tokens(self, index):
        pass

    def collate_fn(self):
        pass

    def batch_sampler(self):
        pass

    def prefetch(self, indices):
        pass

    @property
    def support_prefetch(self):
        return False


class IndexDataset(Seq2seqDataset):
    def __init__(self, data_path: str, dictionary: Dictionary):
        super(IndexDataset, self).__init__(data_path, dictionary)
        self._do_init(data_path)

    def __getitem__(self, index):
        pos = self.pointer[index]
        self.f_bin.seek(pos * 8)
        return np.frombuffer(self.f_bin.read(self.sent_lengths[index] * 8), dtype=np.int64)

    def _do_init(self, data_path):
        with open(data_path + '.ptr', 'rb') as f:
            self.sent_lengths = np.frombuffer(f.read(), dtype=np.int64)
            # block pointer should start from zero
            self.pointer = np.concatenate((np.array([0], dtype=self.sent_lengths.dtype), self.sent_lengths.cumsum()))
        self.f_bin = open(data_path + '.bin', 'rb')

    def num_tokens(self, index):
        return self.sent_lengths[index]

    def batch_sampler(self, max_sentences=None, max_tokens=4096):
        batches = single_dataset_batch(self.sent_lengths, max_tokens, max_sentences, self.num_tokens)
        return batches

    @staticmethod
    def create_index_files(index_path: str, corpus_path: str, dictionary: Dictionary):
        with open(index_path + '.bin', 'wb') as f_bin:
            with open(index_path + '.ptr', 'wb') as f_ptr:
                with open(corpus_path, 'r', encoding='utf-8') as f_data:
                    length = []
                    for line in f_data:
                        line = line.strip().split()
                        line = dictionary.encode_sentence(line)
                        length.append(len(line))
                        f_bin.write(np.array(line, dtype=np.int64).tobytes())
                    # the length of cum_length is dataset samples + 1 due to the starting position 0
                    f_ptr.write(np.array(length, dtype=np.int64).tobytes())

                    logger.info("create {} index samples".format(len(length)))

    def __len__(self):
        return len(self.sent_lengths)

    def __del__(self):
        self.f_bin.close()

    def __getstate__(self):
        state_dict = {"path": self.data_path, "dictionary": self.dictionary}
        return state_dict

    def __setstate__(self, state_dict):
        self._do_init(state_dict["path"])
        self.dictionary = state_dict["dictionary"]


class CacheIndexDataset(IndexDataset):

    def __init__(self, data_path, dictionary):
        super(CacheIndexDataset, self).__init__(data_path, dictionary)
        self.batches = self.batch_sampler()
        self.cache = {}

    def prefetch(self, indices):
        for index in indices:
            self.cache[index] = self.get_item(index)

    def get_item(self, index):
        pos = self.pointer[index]
        self.f_bin.seek(pos * 8)
        return np.frombuffer(self.f_bin.read(self.sent_lengths[index] * 8), dtype=np.int64)

    def __getitem__(self, index):
        if self.cache.get(index, False):
            data = self.cache[index]
            del data[index]
            return data
        else:
            return self.get_item(index)

    def support_prefetch(self):
        return True


class TextDataset(Seq2seqDataset):
    def __init__(self, data_path: str, dictionary: Dictionary):
        super(TextDataset, self).__init__(data_path, dictionary)

        self.dataset = []
        self.sent_lengths = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                self.dataset.append(line)
                self.sent_lengths.append(len(line))

        self.sent_lengths = np.array(self.sent_lengths)

    def __getitem__(self, index):
        return self.dictionary.encode_sentence(self.dataset[index])

    def batch_sampler(self, max_sentences=None, max_tokens=None):
        batches = single_dataset_batch(self.sent_lengths, max_tokens, max_sentences, self.num_tokens)
        return batches

    def __len__(self):
        return len(self.dataset)


class PairDataset(Dataset):
    """

    """

    def __init__(self, src_dataset, tgt_dataset):
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.shuffle = True

    def __getitem__(self, index):
        src_data = self.src_dataset[index]
        tgt_data = self.tgt_dataset[index]
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "src": torch.tensor(src_data, dtype=torch.int64),
            "tgt": torch.tensor(tgt_data, dtype=torch.int64)
        }

    def __len__(self):
        return len(self.src_dataset)

    def num_tokens(self, index):
        return max(self.src_dataset.sent_lengths[index], self.tgt_dataset.sent_lengths[index])

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = np.maximum(self.tgt_dataset.sent_lengths[indices], self.src_dataset.sent_lengths[indices])
        return sizes

    def batch_sampler(self, max_sentences=-1, max_tokens=None) -> List:

        # indices = self.ordered_indices()
        # num_tokens_vec = self.num_tokens_vec(indices)
        # batches = batch_by_size_vec(indices, num_tokens_vec, max_tokens, max_sentences, 8)

        # --- code by wang --- #
        # sorted by target length, then source length

        ordered_indices = self.ordered_indices()

        # length_upper_limit = np.max(self.src_dataset.length[ordered_indices], self.tgt_dataset.length[ordered_indices])

        batches = batch_by_size(indices=ordered_indices,
                                num_tokens_fn=self.num_tokens,
                                max_sentences=max_sentences,
                                max_tokens=max_tokens,
                                required_batch_size_multiple=8)
        # --- code by wang --- #

        return batches

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        # sort by target length, then source length
        indices = indices[np.argsort(self.tgt_dataset.sent_lengths[indices], kind="mergesort")]  # noqa
        ordered_indices = indices[np.argsort(self.src_dataset.sent_lengths[indices], kind="mergesort")]  # noqa
        return ordered_indices

    def collate_fn(self, samples: List[Tuple[List, List]]) -> Dict:
        bsz = len(samples)
        id, src_len, tgt_len = [], [], []

        src_max_len = max(sample["src"].size(0) for sample in samples)  # noqa
        tgt_max_len = max(sample["tgt"].size(0) for sample in samples)  # noqa

        # src_tokens, tgt_tokens, previous_tgt_tokens initialize
        src_tokens = torch.zeros(size=(bsz, src_max_len), dtype=torch.int64).fill_(
            self.src_dataset.dictionary.padding_id)
        tgt_tokens = torch.zeros(size=(bsz, tgt_max_len), dtype=torch.int64).fill_(
            self.tgt_dataset.dictionary.padding_id)
        previous_tgt_tokens = torch.zeros(size=(bsz, tgt_max_len), dtype=torch.int64).fill_(
            self.tgt_dataset.dictionary.padding_id)

        # TODO: check if the assignment operation in-place or not
        for i, sample in enumerate(samples):
            index, src, tgt = sample["index"], sample["src"], sample["tgt"]  # noqa

            # src_tokens[i][src_max_len - len(src):] = src  # left padding
            src_tokens[i][: len(src)] = src
            tgt_tokens[i][: len(tgt)] = tgt
            # eos at the beginning of previous_tgt_tokens, but why?
            previous_tgt_tokens[i][: len(tgt)] = tgt

            id.append(index)
            src_len.append(len(src))
            tgt_len.append(len(tgt))

        src_len, ordered_index = torch.LongTensor(src_len).sort(descending=True)

        id = torch.LongTensor(id).index_select(dim=0, index=ordered_index)
        src_tokens = src_tokens.index_select(dim=0, index=ordered_index)
        tgt_tokens = tgt_tokens.index_select(dim=0, index=ordered_index)
        previous_tgt_tokens = previous_tgt_tokens.index_select(dim=0, index=ordered_index)

        src_masks = (src_tokens == self.src_dataset.dictionary.padding_id)
        tgt_masks = (tgt_tokens == self.tgt_dataset.dictionary.padding_id)
        tgt_lengths = (tgt_tokens != self.tgt_dataset.dictionary.padding_id).sum(-1)
        previous_tgt_masks = (previous_tgt_tokens == self.tgt_dataset.dictionary.padding_id)

        ntokens = tgt_tokens.ne(self.tgt_dataset.dictionary.padding_id).sum()
        #
        # batch = {
        #     "id": id,
        #     "nsentences": torch.tensor(len(samples), dtype=torch.int64),
        #     "ntokens": ntokens,
        #     "net_input": {"src_tokens": src_tokens,
        #                   "src_lengths": src_len,
        #                   "src_masks": src_masks,
        #                   "prev_tgt_tokens": previous_tgt_tokens,
        #                   "prev_tgt_masks": previous_tgt_masks},
        #     "target": tgt_tokens,
        #     "target_masks": tgt_masks
        # }
        batch = {
            "id": id,
            "nsentences": torch.tensor(len(samples), dtype=torch.int64),
            "ntokens": ntokens,
            "net_input": {"src_tokens": src_tokens,
                          "src_lengths": src_len,
                          "prev_tgt_tokens": previous_tgt_tokens},
            "target": tgt_tokens,
            "tgt_lengths": tgt_lengths
        }
        return batch

    def __getstate__(self):
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state


if __name__ == '__main__':
    d_en = Dictionary.load_dictionary_from_file('../test/dict.en')
    d_de = Dictionary.load_dictionary_from_file('../test/dict.de')

    en_dataset = IndexDataset('../test/test.en', max_tokens=4096)
    de_dataset = IndexDataset('../test/test.de', max_tokens=4096)

    dataset = PairDataset(en_dataset, de_dataset)
