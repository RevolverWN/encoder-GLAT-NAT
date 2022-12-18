import torch
from torch.nn import functional as F
INF = 1e10


def softmax(x, T=1):
    return F.softmax(x / T, dim=-1)

    # if x.dim() == 3:
    #     return F.softmax(x.transpose(0, 2)).transpose(0, 2)
    # return F.softmax(x)


def linear_attention(source_masks, decoder_masks, decoder_input_how):
    if decoder_input_how == "copy":
        max_src_len = source_masks.size(1)
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).float() - 1  # batch_size
        trg_lens = decoder_masks.sum(-1).float() - 1  # batch_size
        steps = src_lens / trg_lens  # batch_size

        index_s = torch.arange(max_trg_len)  # max_trg_len
        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        index_s = steps[:, None] * index_s[None, :]  # batch_size X max_trg_len
        index_s = torch.round(index_s).long().detach()
        return index_s

    elif decoder_input_how == "wrap":
        batch_size, max_src_len = source_masks.size()
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).int()  # batch_size

        index_s = torch.arange(max_trg_len)[None, :]  # max_trg_len
        index_s = index_s.repeat(batch_size, 1)  # (batch_size, max_trg_len)

        for sin in range(batch_size):
            if src_lens[sin] + 1 < max_trg_len:
                index_s[sin, src_lens[sin]:2 * src_lens[sin]] = index_s[sin, :src_lens[sin]]

        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        index_s = torch.round(index_s).long().detach()
        return index_s

    elif decoder_input_how == "pad":
        batch_size, max_src_len = source_masks.size()
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).int() - 1  # batch_size

        index_s = torch.arange(max_trg_len)[None, :]  # max_trg_len
        index_s = index_s.repeat(batch_size, 1)  # (batch_size, max_trg_len)

        for sin in range(batch_size):
            if src_lens[sin] + 1 < max_trg_len:
                index_s[sin, src_lens[sin] + 1:] = index_s[sin, src_lens[sin]]

        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        index_s = torch.round(index_s).long().detach()
        return index_s

    elif decoder_input_how == "interpolate":
        max_src_len = source_masks.size(1)
        max_trg_len = decoder_masks.size(1)
        src_lens = source_masks.sum(-1).float()  # batchsize
        trg_lens = decoder_masks.sum(-1).float()  # batchsize
        steps = src_lens / trg_lens  # batchsize
        index_t = torch.arange(0, max_trg_len)  # max_trg_len
        if decoder_masks.is_cuda:
            index_t = index_t.cuda(decoder_masks.get_device())

        index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
        index_s = torch.arange(0, max_src_len)  # max_src_len
        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        indexxx_ = (index_s[None, None, :] - index_t[:, :, None]) ** 2  # batch x max_trg x max_src
        indexxx = softmax(Variable(-indexxx_.float() / 0.3 - INF * (1 - source_masks[:, None, :].float())))  # batch x max_trg x max_src
        return indexxx