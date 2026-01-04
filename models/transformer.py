# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_enc_layers=6,
                 num_dec_layers=6, dim_feed=2048, dropout=0.1,
                 act="relu", norm_before=False,
                 ret_inter=False):
        super().__init__()

        enc_layer = TransformerEncLayer(d_model, nhead, dim_feed,
                                                dropout, act, norm_before)
        enc_norm = nn.LayerNorm(d_model) if norm_before else None
        self.encoder = TransformerEnc(enc_layer, num_enc_layers, enc_norm)

        dec_layer = TransformerDecLayer(d_model, nhead, dim_feed,
                                                dropout, act, norm_before)
        dec_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDec(dec_layer, num_dec_layers, dec_norm,
                                          ret_inter=ret_inter)

        self._reset_params()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, q_emb, y_idx):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        y_emb = q_emb[y_idx].permute(1,0,2)

        tgt = torch.zeros_like(y_emb)
        mem = self.encoder(src)
        hs = self.decoder(tgt, mem, query_pos=y_emb)
                        
        return torch.cat([hs.transpose(1, 2)[-1], y_emb.permute(1,0,2)], -1)


class TransformerEnc(nn.Module):

    def __init__(self, enc_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(enc_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                key_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        out = src

        for layer in self.layers:
            out = layer(out, src_mask=mask,
                           src_key_padding_mask=key_mask, pos=pos)

        if self.norm is not None:
            out = self.norm(out)

        return out


class TransformerDec(nn.Module):

    def __init__(self, dec_layer, num_layers, norm=None, ret_inter=False):
        super().__init__()
        self.layers = _get_clones(dec_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.ret_inter = ret_inter

    def forward(self, tgt, mem,
                tgt_mask: Optional[Tensor] = None,
                mem_mask: Optional[Tensor] = None,
                tgt_key_mask: Optional[Tensor] = None,
                mem_key_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                q_pos: Optional[Tensor] = None):
        out = tgt

        inter = []

        for layer in self.layers:
            out = layer(out, mem, tgt_mask=tgt_mask,
                           mem_mask=mem_mask,
                           tgt_key_mask=tgt_key_mask,
                           mem_key_mask=mem_key_mask,
                           pos=pos, q_pos=q_pos)
            if self.ret_inter:
                inter.append(self.norm(out))

        if self.norm is not None:
            out = self.norm(out)
            if self.ret_inter:
                inter.pop()
                inter.append(out)

        if self.ret_inter:
            return torch.stack(inter)

        return out.unsqueeze(0)


class TransformerEncLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feed=2048, dropout=0.1,
                 act="relu", norm_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.lin1 = nn.Linear(d_model, dim_feed)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_feed, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.act = _get_act_fn(act)
        self.norm_before = norm_before

    def with_pos(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     key_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=key_mask)[0]
        src = src + self.drop1(src2)
        src = self.norm1(src)
        src2 = self.lin2(self.dropout(self.act(self.lin1(src))))
        src = src + self.drop2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    key_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=key_mask)[0]
        src = src + self.drop1(src2)
        src2 = self.norm2(src)
        src2 = self.lin2(self.dropout(self.act(self.lin1(src2))))
        src = src + self.drop2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                key_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.norm_before:
            return self.forward_pre(src, src_mask, key_mask, pos)
        return self.forward_post(src, src_mask, key_mask, pos)


class TransformerDecLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feed=2048, dropout=0.1,
                 act="relu", norm_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mh_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.lin1 = nn.Linear(d_model, dim_feed)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_feed, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        self.act = _get_act_fn(act)
        self.norm_before = norm_before

    def with_pos(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, mem,
                     tgt_mask: Optional[Tensor] = None,
                     mem_mask: Optional[Tensor] = None,
                     tgt_key_mask: Optional[Tensor] = None,
                     mem_key_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     q_pos: Optional[Tensor] = None):
        q = k = self.with_pos(tgt, q_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_mask)[0]
        tgt = tgt + self.drop1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.mh_attn(query=self.with_pos(tgt, q_pos),
                                   key=self.with_pos(mem, pos),
                                   value=mem, attn_mask=mem_mask,
                                   key_padding_mask=mem_key_mask)[0]
        tgt = tgt + self.drop2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.lin2(self.dropout(self.act(self.lin1(tgt))))
        tgt = tgt + self.drop3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, mem,
                    tgt_mask: Optional[Tensor] = None,
                    mem_mask: Optional[Tensor] = None,
                    tgt_key_mask: Optional[Tensor] = None,
                    mem_key_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    q_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos(tgt2, q_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_mask)[0]
        tgt = tgt + self.drop1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.mh_attn(query=self.with_pos(tgt2, q_pos),
                                   key=self.with_pos(mem, pos),
                                   value=mem, attn_mask=mem_mask,
                                   key_padding_mask=mem_key_mask)[0]
        tgt = tgt + self.drop2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.lin2(self.dropout(self.act(self.lin1(tgt2))))
        tgt = tgt + self.drop3(tgt2)
        return tgt

    def forward(self, tgt, mem,
                tgt_mask: Optional[Tensor] = None,
                mem_mask: Optional[Tensor] = None,
                tgt_key_mask: Optional[Tensor] = None,
                mem_key_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                q_pos: Optional[Tensor] = None):
        if self.norm_before:
            return self.forward_pre(tgt, mem, tgt_mask, mem_mask,
                                    tgt_key_mask, mem_key_mask, pos, q_pos)
        return self.forward_post(tgt, mem, tgt_mask, mem_mask,
                                 tgt_key_mask, mem_key_mask, pos, q_pos)


def _get_clones(mod, N):
    return nn.ModuleList([copy.deepcopy(mod) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feed=args.dim_feed,
        num_enc_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        norm_before=args.pre_norm,
        ret_inter=True,
    )


def _get_act_fn(act):
    """Return an activation function given a string"""
    if act == "relu":
        return F.relu
    if act == "gelu":
        return F.gelu
    if act == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {act}.")