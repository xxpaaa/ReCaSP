import copy
from typing import Optional, List
import pickle as cp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        T, B, C = memory.shape
        intermediate = []
        attention_ws = []
        for n, layer in enumerate(self.layers):
            residual = True
            output, ws = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, residual=residual)
            attention_ws.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, attention_ws

class TransformerDecoderWoSelfAttenLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) #可注释掉， 20240912
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) #可注释掉， 20240912
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) #可注释掉， 20240912
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     residual=True):
        # tgt: text_features, torch.Size([14, 1, 768])
        # memory: image_features, torch.Size([49, 1, 768])
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2,ws = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)
        # tgt = self.norm1(tgt)
        tgt2, ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       need_weights=True,
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)

        # attn_weights [B,NUM_Q,T]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, ws

    # @get_local('attn_weights')
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # tgt2 = self.norm1(tgt) # global embedding for self-attention
        # q = k = self.with_pos_embed(tgt2, query_pos)  ## add position embedding
        # tgt2,ws = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask) # self-attention
        # # print('self atten',ws.shape)
        # tgt = tgt + self.dropout1(tgt2) # residual connection

        tgt2 = self.norm2(tgt)  # normalize before multi-head attention for query
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                                 key=self.with_pos_embed(memory, pos),
                                                 value=memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)  # (N,num_heads,L,S) # global to query, local to key, value
        # print('self attn_weights',attn_weights.shape)
        tgt = tgt + self.dropout2(tgt2)  # residual connection
        tgt2 = self.norm3(tgt)  # normalize before feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_weights

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                residual=True):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, residual)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TQN_Model(nn.Module):
    def __init__(self, embed_dim=768, class_num=1, decoder_number_layer=4):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layer = TransformerDecoderWoSelfAttenLayer(self.d_model, 4, 1024,
                                                           0.1, 'relu', normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, decoder_number_layer, self.decoder_norm,
                                          return_intermediate=False)
        self.dropout_feas = nn.Dropout(0.1)

        # #v0
        # self.mlp_head = nn.Sequential(  # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, class_num)
        # )

        self.mlp_head = nn.Sequential(  # nn.LayerNorm(768),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, class_num)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, image_features, text_features, pos=None, return_atten=False, inside_repeat=True):
        # image_features (batch_size, patch_num,dim)
        # text_features (query_num,dim)
        batch_size = image_features.shape[0]
        image_features = image_features.transpose(0, 1)  # (patch_num,batch_size,dim)
        if inside_repeat:
            text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)  # (query_num,batch_size,dim)
        image_features = self.decoder_norm(image_features)
        text_features = self.decoder_norm(text_features)
        features, atten_map = self.decoder(text_features, image_features, memory_key_padding_mask=None, pos=pos, query_pos=None)
        features = self.dropout_feas(features).transpose(0, 1)  # b,embed_dim
        out = self.mlp_head(features)  # (batch_size, query_num)
        if return_atten:
            return out, atten_map
        else:
            return out
