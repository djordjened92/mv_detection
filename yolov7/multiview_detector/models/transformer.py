# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def create_pos_embedding(img_size, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi
    H, W = img_size
    not_mask = torch.ones([1, H, W])
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos

class TransformerWorldFeat(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, dropout=0.1, nhead=8, dim_feedforward=512):
        super(TransformerWorldFeat, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(base_dim * num_cam, hidden_dim, 3, 2, 1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(), )

        self.pos_embedding = create_pos_embedding(np.ceil(np.array(Rworld_shape) / 4).astype(int),
                                                  hidden_dim // 2)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, dropout=dropout, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layer, 3)

        self.upsample = nn.Sequential(nn.Upsample(np.ceil(np.array(Rworld_shape) / 2).astype(int).tolist(),
                                                  mode='bilinear'),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(),
                                      nn.Upsample(Rworld_shape, mode='bilinear'),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(), )

    def forward(self, x, visualize=False):
        B, N, C, H, W = x.shape
        # _, _, H, W = x2.shape
        x = self.downsample(x.view(B, N * C, H, W))
        _, _, H, W = x.shape
        # H*W,B,C*N
        pos_embedding = self.pos_embedding.repeat(B, 1, 1, 1).flatten(2).permute(2, 0, 1).to(x.device)
        x = self.encoder(x.flatten(2).permute(2, 0, 1), pos=pos_embedding)
        merged_feat = self.upsample(x.permute(1, 2, 0).view(B, C, H, W))
        return merged_feat