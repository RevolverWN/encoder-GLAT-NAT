from collections import namedtuple

from fairseq.modules import PositionalEmbedding
from torch import nn

from dictionary import Dictionary
from models import register_model
from models.model_utils import init_bert_params
from models.nat_model import NATdecoder, NAT, default_dict, NATencoder
from modules.pos_attn_transformer import NATTransformerDecoderLayer, NATTransformerDecoder

register_name = "pos_attn_nat_transformer"


@register_model(register_name)
class PosAttnNAT(NAT):
    def __init__(self, config: namedtuple, src_dict: Dictionary, tgt_dict: Dictionary):
        super(NAT, self).__init__()
        self.config = config
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        src_token_emb = nn.Embedding(len(src_dict), self.config.encoder_embed_dim, padding_idx=src_dict.padding_id)
        if config.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if config.encoder_embed_dim != config.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )

            tgt_token_emb = src_token_emb
        else:
            tgt_token_emb = nn.Embedding(len(tgt_dict), self.config.decoder_embed_dim, padding_idx=tgt_dict.padding_id)

        self.encoder = NATencoder(self.config, src_dict, src_token_emb)
        self.decoder = PosAttnNATdecoder(self.config, tgt_dict, tgt_token_emb)
        if config.apply_bert_init:
            self.encoder.apply(init_bert_params)
            self.decoder.apply(init_bert_params)


class PosAttnNATdecoder(NATdecoder):
    def __init__(self, config, dictionary, token_emb):
        super(NATdecoder, self).__init__()
        self.config = config
        self.token_emb = token_emb

        self.position_emb = PositionalEmbedding(num_embeddings=self.config.decoder_max_source_positions,
                                                embedding_dim=self.config.decoder_embed_dim,
                                                padding_idx=dictionary.padding_id)

        self.layer = NATTransformerDecoderLayer(d_model=self.config.decoder_embed_dim,
                                                nhead=self.config.decoder_attention_heads,
                                                batch_first=True)

        self.decoder = NATTransformerDecoder(decoder_layer=self.layer, num_layers=self.config.decoder_layers)

        self.output_layer = nn.Linear(self.token_emb.weight.size(1), self.token_emb.weight.size(0), bias=False)
        self.output_layer.weight = self.token_emb.weight

        self.length_predictor = nn.Linear(self.config.decoder_embed_dim, 256)

    def forward(self, prev_tgt_tokens, prev_tgt_key_padding_masks, encoder_outputs: dict):
        src_emb = encoder_outputs["src_emb"]
        src_masks = encoder_outputs["src_masks"]
        encoder_features = encoder_outputs["encoder_features"]

        # prev_tgt_tokens, prev_tgt_key_padding_masks may have length parallel, so the first dim is
        # batch_size * length_beam_size
        src_batch_size, src_len, src_feat_num = src_emb.size()
        tgt_batch_length_size = prev_tgt_tokens.size(0)
        if src_batch_size != tgt_batch_length_size:
            beam_size = int(tgt_batch_length_size / src_batch_size)
            src_emb = src_emb.unsqueeze(1).repeat(1, beam_size, 1, 1).view(src_batch_size * beam_size, src_len,
                                                                           src_feat_num)
            src_masks = src_masks.unsqueeze(1).repeat(1, beam_size, 1).view(src_batch_size * beam_size, -1)
            encoder_features = encoder_features.unsqueeze(1).repeat(1, beam_size, 1, 1).view(src_batch_size * beam_size,
                                                                                             src_len, src_feat_num)

        if self.config.src_embedding_copy:
            tgt_token_embed = self.copy_src_embed(src_emb, src_masks, prev_tgt_key_padding_masks)
        else:
            tgt_token_embed = self.token_emb(prev_tgt_tokens)

        position_emb = self.position_emb(prev_tgt_tokens)
        true_tgt_embed = self.token_emb(prev_tgt_tokens) + position_emb
        tgt_emb = tgt_token_embed + position_emb

        features = self.decoder(tgt=tgt_emb, memory=encoder_features, memory_key_padding_mask=src_masks,
                                tgt_key_padding_mask=prev_tgt_key_padding_masks, pos_embedding=position_emb)
        logits = self.output_layer(features)

        decoder_outputs = {"logits": logits,
                           "features": features,
                           "tgt_emb": tgt_emb,
                           "true_tgt_embed": true_tgt_embed}
        return decoder_outputs
