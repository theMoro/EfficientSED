"""
minGRU paper: Were RNNs All We Needed? https://arxiv.org/abs/2410.01201v1
Adapted code from Phil Wang's repository at https://github.com/lucidrains/minGRU-pytorch to include a
bidirectional minGRU (under MIT License).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Linear, Identity, Module
from models.sequence_models.helper import RMSNorm


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )


class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.)
        x = self.net(x)
        return x.transpose(1, 2)  # b d n -> b n d


class MinGRUWrapper(Module):
    def __init__(
            self,
            dim,
            depth,
            ff_mult=4,
            min_gru_expansion=1.5,
            conv_kernel_size=3,
            enable_conv=False,
            bidirectional=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size) if enable_conv else None,
                RMSNorm(dim),
                BidirectionalMinGRU(dim, expansion_factor=min_gru_expansion) if bidirectional else MinGRU(dim, expansion_factor=min_gru_expansion),
                RMSNorm(dim),
                FeedForward(dim, mult=ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        for conv, norm, mingru, ff_norm, ff in self.layers:

            # conv
            if exists(conv):
                x = conv(x) + x

            # min gru
            min_gru_out = mingru(
                norm(x)
            )

            x = min_gru_out + x

            # feedforward
            x = ff(ff_norm(x)) + x

        embed = self.norm(x)

        return embed


# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    # Assuming that prev_hidden is provided (which it is not in the usual case)
    # log_coeffs = [0, log(1 - z_1), ..., log(1 - z_T)]  ... forget factors --> a_t
    # log_values = [log(h_0), log(z_1 * \tilde{h}_1), ..., log(z_T * \tilde{h}_T)]  ... gated hidden states  --> b_t

    # a_star[:, t, :] = sum_{i=0}^t log(1 - z_i) = log(prod_{i=1}^t (1 - z_i))
    # (where log(1 - z_0) = 0)
    a_star = log_coeffs.cumsum(dim=1)

    # shifting the log values
    # this subtracts (subtracting in the log-space equals diving in the non log-space) the cumulative log product from
    # the log values, essentially "normalizing" each term relative to the product of the forget factors up to that time
    c = log_values - a_star

    # compute cumulative sum in the log-space --> log(sum_{i=0}^t exp(c_i))
    log_h0_plus_b_star = c.logcumsumexp(dim=1)

    # by adding back the cumulative log product, the function recovers the log of the hidden state log(h_t)
    log_h = a_star + log_h0_plus_b_star

    # take the exponent of log_h to get the hidden state in the normal (non-log) domain (h_t)
    return log_h.exp()


# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class MinGRU(Module):
    def __init__(self, dim, expansion_factor=1., proj_out=None, sequential_mode=False):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        # proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias=False)
        self.to_out = Linear(dim_inner, dim, bias=False) if proj_out else Identity()

        self.sequential_mode = sequential_mode

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        seq_len = x.shape[1]

        # get candidate hidden state and gate values z
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if self.sequential_mode:
            # handle sequential

            hidden = g(hidden)

            gate = gate.sigmoid()

            current_hidden = prev_hidden if exists(prev_hidden) else torch.zeros_like(hidden[:, 0, :])
            outputs = []

            for t in range(seq_len):
                hidden_t = hidden[:, t, :]
                gate_t = gate[:, t, :]

                out_t = current_hidden * (1 - gate_t) + hidden_t * gate_t

                outputs.append(out_t.unsqueeze(1))
                current_hidden = out_t

            out = torch.cat(outputs, dim=1)
        else:
            # parallel

            # Note: -softplus(x) = log(1 - sigmoid(x))
            # Here, -softplus(x) is a numerically stable way of computing log(1-sigmoid(x)).

            # Also, z = sigmoid(gate).

            # Log Forget Factors:
            # log_coeffs = log(1-z) = log(1-sigmoid(gate))    at each time step
            log_coeffs = -F.softplus(gate)

            # Log Update Gate:
            # log_z = log(z) = log(sigmoid(gate))   at each time step
            log_z = -F.softplus(-gate)

            # Log Candidate Hidden State:
            # log_tilde_h = log(g(hidden)) = log(g(W_h * x))    at each time step
            log_tilde_h = log_g(hidden)

            # log_values = log(sigmoid(gate)) + log(g(hidden) = log( sigmoid(gate) * g(hidden) ) = log(z * \tilde{h})
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)

            # extract the hidden states and discard the initial state's direct output (corresponding to prev_hidden)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden


class BidirectionalMinGRU(Module):
    def __init__(self, dim, expansion_factor=1., sequential_mode=False):
        super().__init__()

        self.forward_minGRU = MinGRU(dim, expansion_factor, sequential_mode=sequential_mode)
        self.backward_minGRU = MinGRU(dim, expansion_factor, sequential_mode=sequential_mode)

        dim_inner = int(dim * expansion_factor)
        self.to_out = Linear(dim_inner * 2, dim_inner, bias=False)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        if prev_hidden is None:
            prev_hidden_forward = None
            prev_hidden_backward = None
        else:
            prev_hidden_forward, prev_hidden_backward = prev_hidden

        # Forward pass
        forward_out = self.forward_minGRU(x, prev_hidden_forward)

        # Backward pass
        x_reversed = torch.flip(x, [1])  # reverse the sequence
        backward_out_before_reversing, backward_next_hidden = self.backward_minGRU(x_reversed, prev_hidden_backward, return_next_prev_hidden=True)

        backward_out = torch.flip(backward_out_before_reversing, [1])  # reverse back to original order

        # Combine outputs
        combined_out = torch.concat([forward_out, backward_out], dim=-1)

        # Project to output dimension (as if we had only one MinGRU / direction)
        combined_out = self.to_out(combined_out)

        if not return_next_prev_hidden:
            return combined_out

        # Get next hidden states
        forward_next_hidden = forward_out[:, -1:]

        return combined_out, (forward_next_hidden, backward_next_hidden)
