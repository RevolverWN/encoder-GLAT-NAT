import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, query_size, context_size, output_size):
        super(Attention, self).__init__()
        self.proj_in = nn.Linear(query_size, context_size)
        self.proj_out = nn.Linear(query_size + context_size, output_size)

    def forward(self, query, context, context_mask):
        """

        :param context_mask:
        :param query: [bsz, query_size]
        :param context: [bsz, seq_len, context_size]
        """
        query = self.proj_in(query).unsqueeze(1)
        scores = (query * context).sum(2)
        if context_mask:
            scores.masked_fill_(context_mask, -np.inf)
        weight = F.softmax(scores, dim=-1).unsqueeze(2)
        context_vector = (weight * context).sum(1)
        result = torch.tanh(self.proj_out(torch.cat((query, context_vector), dim=1)))

        return result

