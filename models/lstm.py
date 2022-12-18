from collections import namedtuple
from typing import List

import torch

from models import register_model
import torch.nn as nn

register_name = "lstm"
default_dict = {}

class LstmEncoder(nn.Module):
    def __init__(self, src_dict):
        super(LstmEncoder, self).__init__()
        self.src_dict = src_dict
        self.src_embed = nn.Embedding(num_embeddings=len(self.src_dict),
                                      embedding_dim=config.emb_dim,
                                      padding_idx=self.src_dict.padding)
        self.encoder_model = nn.LSTM(input_size=config.emb_dim,
                                     hidden_size=config.hidden_size,
                                     batch_first=True,
                                     num_layers=config.encoder_num_layers,
                                     bidirectional=config.bidirectional)

    def forward(self, src_tokens, length: List, src_mask):
        """
        :param src_mask:
        :param length: [bsz]
        :param src_tokens: [bsz, seq_len] padded
        """
        bsz, seq_len = src_tokens.size()
        length_desc = torch.tensor(reversed(length))
        emb = self.src_embed(src_tokens[length_desc])
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input=emb,
                                                               lengths=length_desc,
                                                               batch_first=True)
        output, (h_n, c_n) = self.encoder_model(packed_input)

        output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # output -> [bsz, seq_len, {2}*hidden_size]
        # h_n -> [{2}*num_layers ,bsz, hidden_size]
        # c_n -> [{2}*num_layers ,bsz, hidden_size]
        # src_mask -> [bsz, seq_len]
        # we change h_n and c_n dimension to [num_layers, bsz, {2}*hidden_size] for decoder 0 timestep h_0, c_0
        transmission_line = 2 if config.bidirectional else 1
        h_n = h_n.view(config.encoder_num_layers, transmission_line, bsz, config.hidden_size).transpose(1, 2)
        h_n = h_n.view(config.encoder_num_layers, bsz, transmission_line*config.hidden_size)
        c_n = c_n.view(config.encoder_num_layers, transmission_line, bsz, config.hidden_size).transpose(1, 2)
        c_n = c_n.view(config.encoder_num_layers, bsz, transmission_line * config.hidden_size)

        encoder_output = {'output': output, 'h_n': h_n, 'c_n': c_n, 'src_mask': src_mask}
        return encoder_output  # [bsz, seq_len, hidden_size]


class LstmDecoder(nn.Module):
    def __init__(self, tgt_dict):
        super(LstmDecoder, self).__init__()
        self.tgt_dict = tgt_dict
        transmission_line = 2 if config.bidirectional else 1

        self.emb = nn.Embedding(num_embeddings=len(self.tgt_dict),
                                embedding_dim=config.emb_dim,
                                padding_idx=self.tgt_dict.padding_idx)
        self.encoder_hidden_proj = nn.Linear(2 * config.hidden_size,
                                             config.hidden_size) if config.bidirectional else None
        self.encoder_cell_proj = nn.Linear(2 * config.hidden_size,
                                             config.hidden_size) if config.bidirectional else None

        self.layers = nn.ModuleList([nn.LSTMCell(
            input_size=self.emb.embedding_dim + config.hidden_size
            if layer == 0 else config.hidden_size,
            hidden_size=config.hidden_size)
            for layer in range(config.decoder_num_layers)])

        self.attention = Attention(config.hidden_size, transmission_line * config.hidden_size, config.hidden_size)

        self.projection = nn.Linear(config.hidden_size, len(self.tgt_dict))

    def forward(self, encoder_output, previous_tgt_tokens=None):
        """

        :param previous_tgt_tokens: [bsz, seq_len] padded
        """
        bsz = encoder_output['output'][0]
        previous_h_x = torch.unbind(encoder_output['h_n'])
        previous_c_x = torch.unbind(encoder_output['c_n'])
        if config.bidirectional:
            previous_h_x = [self.encoder_hidden_proj(h) for h in previous_h_x]
            previous_c_x = [self.encoder_cell_proj(c) for c in previous_c_x]

        previous_final_hidden = torch.zeros(size=(bsz, config.hidden_size))
        output = []

        # train mode: teacher forcing
        if previous_tgt_tokens:
            seq_len = previous_tgt_tokens.size(1)

            tgt_embedding = self.emb(previous_tgt_tokens)
            for time_step in range(seq_len):
                previous_h_x, previous_c_x, previous_final_hidden = self.step_forward(tgt_embedding,
                                                                                      time_step,
                                                                                      previous_final_hidden,
                                                                                      previous_h_x,
                                                                                      previous_c_x,
                                                                                      encoder_output)
                output.append(previous_final_hidden)

            logits = self.projection(torch.stack([h[-1] for h in output]))
        else:
            pass


    def step_forward(self, tgt_embedding, time_step, previous_final_hidden, previous_h_x, previous_c_x, encoder_output):
        for lay_num, layer in enumerate(self.layers):
            if lay_num == 0:
                h, c = layer(torch.cat((tgt_embedding[:, time_step, :], previous_final_hidden), dim=1),
                             (previous_h_x[lay_num], previous_c_x[lay_num]))
            else:
                h, c = layer(h, (previous_h_x[lay_num], previous_c_x[lay_num]))
            previous_h_x[lay_num] = h
            previous_c_x[lay_num] = c

        previous_final_hidden = self.attention(h, encoder_output['output'], encoder_output['src_mask'])
        return previous_h_x, previous_c_x, previous_final_hidden

@register_model(register_name)
class LstmModel(nn.Module):
    """
    we implement this seq2seq model based on the encoder layers number is equal to decoder layers number.

    """

    config = None
    def __init__(self):
        """

        """
        super(LstmModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder_out = self.encoder(x)