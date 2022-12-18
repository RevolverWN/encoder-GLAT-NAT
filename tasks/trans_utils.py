from typing import List

import torch


def remove_invalid_token(sentences: List[List], invalid_token: List) -> List[List]:
    return [list(filter(lambda x: x not in invalid_token, line)) for line in sentences]


def remove_bpe(sentence: str) -> str:
    return (sentence + " ").replace("@@ ", "").rstrip()


def remove_repeat_token(sentence: str) -> str:
    sentence = sentence.strip().split()
    res = []
    for token in sentence:
        if len(res) == 0:
            res.append(token)
        elif token != res[-1]:
            res.append(token)

    return " ".join(res)


def assign_single_value_long(x, i, y):
    temp = x.clone()
    b, l = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)
    temp.view(-1)[i.view(-1)] = y
    return temp


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_long(x, i, y):
    temp = x.clone()
    b, l = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)
    temp.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]
    return temp
