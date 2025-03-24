"""
Attention with rotary position embedding, adapted from Phil Wang's repository
at https://github.com/lucidrains/BS-RoFormer (under MIT License).
"""

from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from models.sequence_models.helper import RMSNorm

def exists(val):
    return val is not None


class Attend(nn.Module):
    def __init__(self, dropout=0.0, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale

        self.add_zero_kv = False

    def forward(self, q, k, v, attn_bias=None):
        batch, heads, q_len, _ = q.shape

        global first_RUN

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        if self.add_zero_kv:
            k, v = tuple(F.pad(t, (0, 0, 1, 0), value=0.) for t in (k, v))

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value=0.)

        if exists(attn_bias):
            attn_bias = attn_bias.expand(batch, heads, -1, -1)

            if first_RUN:
                print("Using attn_bias (positional encoding)", attn_bias.shape)

        first_RUN = False

        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, attn_mask=attn_bias
        )


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        gating=True,
        pos_encoding_type="rotary"
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.pos_encoding_type = pos_encoding_type
        self.pos_encoding = None

        self.num_mem_kv = 0

        if pos_encoding_type == "rotary":
            self.pos_encoding = RotaryEmbedding(dim_head)
        elif pos_encoding_type != "none":
            raise ValueError(f"Unknown positional encoding: {pos_encoding_type}")

        self.attend = Attend(dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        if gating:
            self.to_gates = nn.Linear(dim, heads)
        else:
            self.to_gates = None

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x, pos=None):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )

        if self.pos_encoding_type == "rotary":
            q = self.pos_encoding.rotate_queries_or_keys(q)
            k = self.pos_encoding.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        if exists(self.to_gates):
            gates = self.to_gates(x)
            out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
