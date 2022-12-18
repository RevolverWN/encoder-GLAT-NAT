from __future__ import annotations
from collections import Counter
from itertools import chain
from typing import List


class Dictionary(object):

    def __init__(self, word2count: dict):
        """

        :param word2count: {token: count}
        """
        self.word2count = word2count

        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.mask = "<MASK>"
        self.cls = "<CLS>"
        self.spec_token = [self.unk, self.pad, self.bos, self.eos, self.mask, self.cls]

        self.word2id = {}
        for index, token in enumerate(chain(self.spec_token, self.word2count.keys())):
            self.word2id[token] = index

        self.id2word = {idx: word for word, idx in self.word2id.items()}

    @classmethod
    def build_dictionary(cls, corpus: List, min_freq: int = 0) -> Dictionary:
        count = Counter()
        word2count = {}

        for path in corpus:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    count.update(line.strip().split())

        for token, c in count.most_common():
            if c >= min_freq:
                word2count[token] = c

        return cls(word2count)

    @classmethod
    def load_dictionary_from_file(cls, filename: str) -> Dictionary:
        """

        :param filename: file lines "token count"
        """
        word2count = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                token, count = line[0], int(line[1])
                word2count[token] = count

        return cls(word2count)

    def save_dict(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for token, count in self.word2count.items():
                f.write(token + ' ' + str(count) +'\n')

    def __getitem__(self, token: str) -> int:
        return self.word2id[token]

    def encode_sentence(self, sentence: List[str]) -> List[int]:
        res = []
        for token in sentence:
            if token in self.word2id:
                res.append(self.word2id[token])
            else:
                res.append(self.word2id[self.unk])
        return res

    def decode_sentence(self, sentence: List[int]) -> str:
        return ' '.join([self.id2word[idx] for idx in sentence])

    @property
    def padding_id(self) -> int:
        return self.word2id[self.pad]

    @property
    def unk_id(self) -> int:
        return self.word2id[self.unk]

    @property
    def bos_id(self) -> int:
        return self.word2id[self.bos]

    @property
    def eos_id(self) -> int:
        return self.word2id[self.eos]

    @property
    def mask_id(self) -> int:
        return self.word2id[self.mask]

    @property
    def cls_id(self) -> int:
        return self.word2id[self.cls]

    @property
    def spec_token_id(self) -> List:
        return [self.word2id[token] for token in self.spec_token]

    def __len__(self) -> int:
        return len(self.word2id)

    def __eq__(self, other: Dictionary):
        return self.word2id == other.word2id





