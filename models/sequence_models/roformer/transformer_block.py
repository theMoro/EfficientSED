"""
Transformer Block with rotary position embedding, adapted from Phil Wang's repository
at https://github.com/lucidrains/BS-RoFormer (under MIT License).
"""

from torch import nn
from torch.nn import Module, ModuleList
from models.sequence_models.helper import RMSNorm
from models.sequence_models.roformer.attention import Attention

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
        dim_out=None,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        dim_inner = int(dim * mult)
        self.activation = nn.GELU()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=32,
        heads=16,
        attn_dropout=0.1,
        pos_encoding_type="rotary",
        ff_dropout=0.1,
        ff_mult=4,
        norm_output=True,
        gating=True
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            pos_encoding_type=pos_encoding_type,
                            gating=gating,
                        ),
                        ff,
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.norm(x)
        return x