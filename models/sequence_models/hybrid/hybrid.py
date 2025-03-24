"""
Code adapted from Phil Wang's repository at https://github.com/lucidrains/minGRU-pytorch (under MIT License).
"""

from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from models.sequence_models.hybrid.minGRU import MinGRU, BidirectionalMinGRU
from models.sequence_models.helper import RMSNorm

from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class HybridWrapper(Module):
    def __init__(
            self,
            dim,
            depth,
            ff_mult = 4,
            heads = 8,
            dim_head = 64,
            bidirectional = False
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                RMSNorm(dim),
                MinGRUAttnHybrid(dim, dim_head=dim_head, heads=heads, bidirectional=bidirectional),
                RMSNorm(dim),
                FeedForward(dim, mult=ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        for norm, hybrid, ff_norm, ff in self.layers:
            # hybrid attn + min gru / min lstm
            x = hybrid(norm(x)) + x

            # feedforward
            x = ff(ff_norm(x)) + x

        embed = self.norm(x)

        return embed



# hybrid minGRU and attention, following the same design as Hymba
# Hymba split the features into two, carried out associative scan RNN + attention on separate branches, followed by norm, scale for both then averaged, projected out

class MinGRUAttnHybrid(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        learned_mix = False,
        bidirectional = False,
        dropout=0.0
    ):
        super().__init__()
        self.heads = heads

        if heads * dim_head != dim:
            print("WARNING: heads * dim_head != dim in MinGRUAttnHybrid, using dim_inner > dim now!")

        dim_inner = heads * dim_head

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_mix = nn.Sequential(RMSNorm(dim), nn.Linear(dim, heads, bias = False)) if learned_mix else None

        print("MinGRUAttnHybrid: bidirectional = ", bidirectional)
        print("MinGRUAttnHybrid: expansion_factor = ", dim_inner / dim)

        self.rnn = BidirectionalMinGRU(dim, expansion_factor = dim_inner / dim) \
            if bidirectional else MinGRU(dim, expansion_factor = dim_inner / dim, proj_out = True)

        self.rnn_out_norm = RMSNorm(dim_head)
        self.attn_out_norm = RMSNorm(dim_head)

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

        self.dropout = dropout

    def forward(
        self,
        x
    ):
        # gru branch
        rnn_out = self.rnn(x)  # 64 x 250 x 1024

        # attention branch
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = False
        )

        # rearrange rnn_out
        rnn_out = rearrange(rnn_out, 'b n (h d) -> b h n d', h=self.heads)  # 64 x 8 x 250 x 128

        # in paper, they simply averaged the two branches
        mix = 0.5

        if exists(self.to_mix):
            # maybe learned per-token / head mixing
            mix = self.to_mix(x).sigmoid()
            mix = rearrange(mix, 'b n h -> b h n 1')

        # the scheme for hybridizing is normalizing + scaling each branch then averaging
        out = mix * self.rnn_out_norm(rnn_out) + (1. - mix) * self.attn_out_norm(attn_out)

        out = F.dropout(out, p=self.dropout)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)