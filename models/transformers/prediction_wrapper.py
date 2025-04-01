import os

import torch
import torch.nn as nn
from torch.hub import download_url_to_file
from torch.nn import ModuleList

from config import RESOURCES_FOLDER, CHECKPOINT_URLS
from models.sequence_models.helper import init_weights

# sequence models
# from mamba_ssm import Mamba2

from models.sequence_models.multi_layer_model import MultiLayerModel, MultiLayerModelFp32
from models.sequence_models.tcn.tcn import TemporalConvNet
from models.sequence_models.gru.gru import GRU
from models.sequence_models.hybrid.hybrid import HybridWrapper
from models.sequence_models.roformer.attention import Attention
from models.sequence_models.roformer.transformer_block import TransformerBlock

FIRST_RUN = True


class PredictionsWrapper(nn.Module):
    """
        A wrapper module that adds an optional sequence model and classification heads on top of a base model (CNN).

        Args:
            base_model (nn.Module): The base model (CNN) providing sequence embeddings
            checkpoint (str, optional): checkpoint name for loading pre-trained weights. Default is None.
            n_classes_strong (int): Number of classes for strong predictions. Default is 447.
            n_classes_weak (int, optional): Number of classes for weak predictions. Default is None,
                                            which sets it equal to n_classes_strong.
            embed_dim (int, optional): Embedding dimension of the base model output. Default is 768.
            seq_len (int, optional): Desired sequence length. Default is 250 (40 ms resolution).
            seq_model_type (str, optional): Type of sequence model to use.
                                            Default is None, which means no additional sequence model is used.
            seq_model_layers (int, optional): Number of layers in the sequence model. Default is 2.
            seq_model_dim (int, optional): Dimension of the sequence model. Default is 256.
            head_type (str, optional): Type of classification head. Choices are ["linear", "attention", "None"].

            gru_dropout (float, optional): Dropout rate for GRU layers. Default is 0.0.
            gru_bidirectional (bool, optional): Whether to use bidirectional GRU. Default is True.

            attn_heads (int, optional): Number of attention heads in each self-attention layer. Default is 8.
            attn_head_dim (int, optional): Dimension of each attention head. If None, attn_dim // attn_heads is used. Default is None.
            attn_pos_encoding_type (str, optional): Type of positional encoding to use in self-attention. Default is "rotary".
            attn_drop (float, optional): Dropout rate for self-attention layers. Default is 0.0.
            attn_gating (bool, optional): Whether to use gating in self-attention. Default is True.

            tf_heads (int, optional): Number of attention heads in each transformer layer. Default is 8.
            tf_head_dim (int, optional): Dimension of each attention head. If None, tf_dim // tf_heads is used. Default is None.
            tf_pos_encoding_type (str, optional): Type of positional encoding to use in transformer. Default is "rotary".
            tf_drop (float, optional): Dropout rate for transformer layers. Default is 0.2.
            tf_ff_mult (int, optional): Feedforward multiplier in transformer. Default is 4.
            tf_gating (bool, optional): Whether to use gating in transformer. Default is True.

            mamba_d_state (int, optional): Dimension of the state variable in Mamba model. Default is 64.

            tcn_blocks (int, optional): Number of blocks in TCN. Default is 5.
            tcn_kernel_size (int, optional): Kernel size in TCN. Default is 3.
            tcn_dropout (float, optional): Dropout rate for TCN layers. Default is 0.2.
            tcn_activation_func (str, optional): Activation function in TCN. Default is "relu".

            hybrid_heads (int, optional): Number of attention heads in each hybrid layer. Default is 8.
            hybrid_dim_head (int, optional): Dimension of each attention head. If None, hybrid_dim // hybrid_heads is used. Default is None.
            hybrid_ff_mult (int, optional): Feedforward multiplier in hybrid model. Default is 4.
            hybrid_bidirectional (bool, optional): Whether to use bidirectional attention in hybrid model. Default is True.
        """

    def __init__(self,
                 base_model,
                 checkpoint=None,
                 n_classes_strong=447,
                 n_classes_weak=None,
                 embed_dim=768,
                 seq_len=250,
                 seq_model_type=None,
                 seq_model_layers=2,
                 seq_model_dim=256,
                 head_type="linear",

                 gru_dropout=0.0,
                 gru_bidirectional=True,

                 attn_heads=8,
                 attn_head_dim=None,
                 attn_pos_encoding_type="rotary",
                 attn_drop=0.0,
                 attn_gating=True,

                 tf_heads=8,
                 tf_head_dim=None,
                 tf_pos_encoding_type="rotary",
                 tf_drop=0.2,
                 tf_ff_mult=4,
                 tf_gating=True,

                 mamba_d_state=64,

                 tcn_blocks=5,
                 tcn_kernel_size=3,
                 tcn_dropout=0.2,
                 tcn_activation_func="relu",

                 hybrid_heads=8,
                 hybrid_dim_head=None,
                 hybrid_ff_mult=4,
                 hybrid_bidirectional=True,
                 ):
        super(PredictionsWrapper, self).__init__()
        self.model = base_model
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_classes_strong = n_classes_strong
        self.n_classes_weak = n_classes_weak if n_classes_weak is not None else n_classes_strong
        self.seq_model_type = seq_model_type
        self.head_type = head_type

        # sequence model

        ## transformer blocks
        if self.seq_model_type == "tf":
            tf_dim = seq_model_dim
            tf_layers = seq_model_layers

            assert tf_dim >= 128, "transformer_block might not be implemented correctly for tf_dim < 128"
            assert (
                    tf_dim % tf_heads == 0
            ), "tf_dim must be divisible by tf_heads"

            tf_head_dim = tf_dim // tf_heads if tf_head_dim is None else tf_head_dim
            assert tf_head_dim * tf_heads == tf_dim, "tf_head_dim * tf_heads must be equal to tf_dim"

            self.seq_model = nn.Sequential(
                nn.Linear(embed_dim, tf_dim) if embed_dim != tf_dim else nn.Identity(),
                TransformerBlock(
                    dim=tf_dim,
                    depth=tf_layers,
                    heads=tf_heads,
                    attn_dropout=tf_drop,
                    ff_dropout=tf_drop,
                    pos_encoding_type=tf_pos_encoding_type,
                    ff_mult=tf_ff_mult,
                    dim_head=tf_head_dim,
                    norm_output=True,
                    gating=tf_gating,
                )
            )

            self.seq_model.apply(init_weights)
            num_features = tf_dim

        elif self.seq_model_type == "attn":
            attn_dim = seq_model_dim
            attn_layers = seq_model_layers

            assert attn_dim >= 128, "attention might not be implemented correctly for attn_dim < 128"
            assert (
                    attn_dim % attn_heads == 0
            ), "attn_dim must be divisible by attn_heads"

            attn_head_dim = attn_dim // attn_heads if attn_head_dim is None else attn_head_dim
            assert attn_head_dim * attn_heads == attn_dim, "attn_head_dim * attn_heads must be equal to attn_dim"

            layers = ModuleList([])
            for _ in range(attn_layers):
                layers.append(
                    Attention(
                        dim=attn_dim,
                        dim_head=attn_head_dim,
                        heads=attn_heads,
                        dropout=attn_drop,
                        pos_encoding_type=attn_pos_encoding_type,
                        gating=attn_gating
                    )
                )

            self.seq_model = nn.Sequential(
                nn.Linear(embed_dim, attn_dim) if embed_dim != attn_dim else nn.Identity(),
                MultiLayerModel(layers)
            )
            self.seq_model.apply(init_weights)
            num_features = attn_dim

        elif self.seq_model_type == "gru":
            gru_dim = seq_model_dim
            gru_layers = seq_model_layers

            self.seq_model = nn.Sequential(
                # we have found that adding a linear layer before the GRU decreases performance and leads to more unstable training
                GRU(
                    n_in=embed_dim,
                    n_hidden=gru_dim,
                    dropout=gru_dropout,
                    num_layers=gru_layers,
                    bidirectional=gru_bidirectional
                )
            )

            num_features = gru_dim * 2 if gru_bidirectional else gru_dim

        elif self.seq_model_type == "mamba":
            mamba_dim = seq_model_dim
            mamba_layers = seq_model_layers

            assert mamba_dim >= 128, "Mamba might not be implemented correctly for mamba_dim < 128"

            headdim = 32 if mamba_dim == 128 else 64  # we need to use headdim=32 if mamba_dim=128

            layers = ModuleList([])
            for i in range(mamba_layers):
                layers.append(
                    nn.Sequential(
                        Mamba2(
                            d_model=mamba_dim,
                            d_state=mamba_d_state,
                            d_conv=3,  # has to be between 2 and 4
                            expand=2,
                            headdim=headdim,
                            rmsnorm=False,
                            use_mem_eff_path=False
                        ).to(torch.float32),
                        nn.LayerNorm(mamba_dim)
                    )

                )

            self.seq_model = nn.Sequential(
                nn.Linear(embed_dim, mamba_dim) if embed_dim != mamba_dim else nn.Identity(),
                nn.LayerNorm(mamba_dim),
                MultiLayerModelFp32(layers)  # for Mamba, we need to use fp32 for stable training
            )
            num_features = mamba_dim

        elif self.seq_model_type == "tcn":
            tcn_input_channels = seq_model_dim
            tcn_layers = seq_model_layers

            act_func = nn.ELU if tcn_activation_func == "elu" else nn.ReLU

            layers = ModuleList([])
            for i in range(tcn_layers):
                layers.append(
                    TemporalConvNet(
                        num_inputs=tcn_input_channels,
                        num_channels=[tcn_input_channels] * tcn_blocks,
                        kernel_size=tcn_kernel_size,
                        dropout=tcn_dropout,
                        activation_func=act_func
                    )
                )

            self.seq_model = nn.Sequential(
                nn.Linear(embed_dim, tcn_input_channels) if embed_dim != tcn_input_channels else nn.Identity(),
                MultiLayerModel(layers)
            )

            num_features = tcn_input_channels

        elif self.seq_model_type == "hybrid":
            hybrid_dim = seq_model_dim
            hybrid_layers = seq_model_layers

            assert hybrid_dim >= 128, "Hybrid might not be implemented correctly for hybrid_dim < 128"

            assert (
                    hybrid_dim % hybrid_heads == 0
            ), "hybrid_dim must be divisible by hybrid_heads"

            hybrid_dim_head = hybrid_dim // hybrid_heads if hybrid_dim_head is None else hybrid_dim_head
            assert hybrid_dim_head * hybrid_heads == hybrid_dim, "hybrid_dim_head * hybrid_heads must be equal to hybrid_dim"

            self.seq_model = nn.Sequential(
                nn.Linear(embed_dim, hybrid_dim) if embed_dim != hybrid_dim else nn.Identity(),
                HybridWrapper(
                    dim=hybrid_dim,
                    depth=hybrid_layers,
                    ff_mult=hybrid_ff_mult,
                    heads=hybrid_heads,
                    dim_head=hybrid_dim_head,
                    bidirectional=hybrid_bidirectional
                )
            )

            num_features = hybrid_dim

        elif self.seq_model_type is None:
            self.seq_model = nn.Identity()
            # no additional sequence model
            num_features = self.embed_dim
        else:
            raise ValueError(f"Unknown seq_model_type: {self.seq_model_type}")

        self.num_features = num_features

        if self.head_type == "attention":
            assert self.n_classes_strong == self.n_classes_weak, "head_type=='attention' requires number of strong and " \
                                                                 "weak classes to be the same!"

        if self.head_type is not None:
            self.strong_head = nn.Linear(num_features, self.n_classes_strong)
            self.weak_head = nn.Linear(num_features, self.n_classes_weak)

        if checkpoint is not None:
            print("Loading pretrained checkpoint: ", checkpoint)
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt_file = os.path.join(RESOURCES_FOLDER, checkpoint + ".pt")
        if not os.path.exists(ckpt_file):
            download_url_to_file(CHECKPOINT_URLS[checkpoint], ckpt_file)
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)

        # TODO: add support for loading fmn models
        # compatibility with uniform wrapper structure we introduced for the public repo
        if 'fpasst' in checkpoint:
            state_dict = {("model.fpasst." + k[len("model."):] if k.startswith("model.")
                           else k): v for k, v in state_dict.items()}
        elif 'M2D' in checkpoint:
            state_dict = {("model.m2d." + k[len("model."):] if not k.startswith("model.m2d.") and k.startswith("model.")
                           else k): v for k, v in state_dict.items()}
        elif 'BEATs' in checkpoint:
            state_dict = {("model.beats." + k[len("model.model."):] if k.startswith("model.model.")
                           else k): v for k, v in state_dict.items()}
        elif 'ASIT' in checkpoint:
            state_dict = {("model.asit." + k[len("model."):] if k.startswith("model.")
                           else k): v for k, v in state_dict.items()}
        elif 'fmn' in checkpoint:
            state_dict = {("model." + k[len("net."):] if k.startswith("net.")
                           else k): v for k, v in state_dict.items()}
            state_dict = {("model.fmn." + k[len("model.model."):] if k.startswith("model.model.")
                           else k): v for k, v in state_dict.items()}
            state_dict = {(k[len("model."):] if not k.startswith("model.fmn.")
                           else k): v for k, v in state_dict.items()}

        n_classes_weak_in_sd = state_dict['weak_head.bias'].shape[0] if 'weak_head.bias' in state_dict else -1
        n_classes_strong_in_sd = state_dict['strong_head.bias'].shape[0] if 'strong_head.bias' in state_dict else -1
        seq_model_in_sd = any(['seq_model.' in key for key in state_dict.keys()])
        keys_to_remove = []
        strict = True
        expected_missing = 0
        if self.head_type is None:
            # remove all keys related to head
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
        elif self.seq_model_type is not None and not seq_model_in_sd:
            # we want to train a sequence model (e.g., gru) on top of a
            #   pre-trained transformer (e.g., AS weak pretrained)
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
            num_seq_model_keys = len([key for key in self.seq_model.state_dict()])
            expected_missing = len(keys_to_remove) + num_seq_model_keys
            strict = False
        else:
            # head type is not None
            if n_classes_weak_in_sd != self.n_classes_weak:
                # remove weak head from sd
                keys_to_remove.append('weak_head.bias')
                keys_to_remove.append('weak_head.weight')
                strict = False
            if n_classes_strong_in_sd != self.n_classes_strong:
                # remove strong head from sd
                keys_to_remove.append('strong_head.bias')
                keys_to_remove.append('strong_head.weight')
                strict = False
            expected_missing = len(keys_to_remove)

        # allow missing mel parameters for compatibility
        num_mel_keys = len([key for key in self.state_dict() if 'mel_transform' in key])
        if num_mel_keys > 0:
            expected_missing += num_mel_keys
            strict = False

        state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        assert len(missing) == expected_missing
        assert len(unexpected) == 0

    def separate_params(self):
        if hasattr(self, "separate_params"):
            return self.model.separate_params()
        else:
            raise NotImplementedError("The base model has no 'separate_params' method!'")

    def has_separate_params(self):
        return hasattr(self.model, "separate_params")

    def mel_forward(self, x):
        return self.model.mel_forward(x)

    def forward(self, x):
        # base model is expected to output a sequence
        # (batch size x sequence length x embedding dimension)
        x = self.model(x)

        # ATST: x.shape: batch size x 250 x 768
        # PaSST: x.shape: batch size x 250 x 768
        # ASiT: x.shape: batch size x 497 x 768
        # M2D: x.shape: batch size x 62 x 3840
        # BEATs: x.shape: batch size x 496 x 768

        assert len(x.shape) == 3

        if x.size(-2) > self.seq_len:
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.seq_len).transpose(1, 2)
        elif x.size(-2) < self.seq_len:
            x = torch.nn.functional.interpolate(x.transpose(1, 2), size=self.seq_len,
                                                mode='linear').transpose(1, 2)

        x = self.seq_model(x)

        if self.head_type == "attention":
            # attention head to obtain weak from strong predictions
            strong = torch.sigmoid(self.strong_head(x))
            sof = torch.softmax(self.weak_head(x), dim=-1)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)
            return strong.transpose(1, 2), weak
        elif self.head_type == "linear":
            # on AudioSet strong, only strong predictions are used
            # on AudioSet weak, only weak predictions are used
            # why both? because we tried to simultaneously train on AudioSet weak and strong (less successful)
            strong = self.strong_head(x)
            weak = self.weak_head(x.mean(dim=1))
            return strong.transpose(1, 2), weak
        else:
            # no head means the sequence is returned instead of strong and weak predictions
            return x
