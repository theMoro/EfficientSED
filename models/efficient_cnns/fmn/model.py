import os
import urllib.parse
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from torchvision.ops.misc import ConvNormActivation

from models.efficient_cnns.fmn.block_types import InvertedResidualConfig, InvertedResidual
from models.efficient_cnns.fmn.utils import cnn_out_size

# Adapted version of MobileNetV3 pytorch implementation
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

# points to github releases
model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
# folder to store downloaded models to
model_dir = "resources"

pretrained_models = {
    # pytorch ImageNet pre-trained model
    # own ImageNet pre-trained models will follow
    # NOTE: for easy loading we provide the adapted state dict ready for AudioSet training (1 input channel,
    # 527 output classes)
    # NOTE: the classifier is just a random initialization, feature extractor (conv layers) is pre-trained
    "mn10_im_pytorch": urllib.parse.urljoin(model_url, "mn10_im_pytorch.pt"),
    # self-trained models on ImageNet
    "mn01_im": urllib.parse.urljoin(model_url, "mn01_im.pt"),
    "mn02_im": urllib.parse.urljoin(model_url, "mn02_im.pt"),
    "mn04_im": urllib.parse.urljoin(model_url, "mn04_im.pt"),
    "mn05_im": urllib.parse.urljoin(model_url, "mn05_im.pt"),
    "mn06_im": urllib.parse.urljoin(model_url, "mn06_im.pt"),
    "mn10_im": urllib.parse.urljoin(model_url, "mn10_im.pt"),
    "mn20_im": urllib.parse.urljoin(model_url, "mn20_im.pt"),
    "mn30_im": urllib.parse.urljoin(model_url, "mn30_im.pt"),
    "mn40_im": urllib.parse.urljoin(model_url, "mn40_im.pt"),
    # Models trained on AudioSet
    "mn01_as": urllib.parse.urljoin(model_url, "mn01_as_mAP_298.pt"),
    "mn02_as": urllib.parse.urljoin(model_url, "mn02_as_mAP_378.pt"),
    "mn04_as": urllib.parse.urljoin(model_url, "mn04_as_mAP_432.pt"),
    "mn05_as": urllib.parse.urljoin(model_url, "mn05_as_mAP_443.pt"),
    "mn10_as": urllib.parse.urljoin(model_url, "mn10_as_mAP_471.pt"),
    "mn20_as": urllib.parse.urljoin(model_url, "mn20_as_mAP_478.pt"),
    "mn30_as": urllib.parse.urljoin(model_url, "mn30_as_mAP_482.pt"),
    "mn40_as": urllib.parse.urljoin(model_url, "mn40_as_mAP_484.pt"),
    "mn40_as(2)": urllib.parse.urljoin(model_url, "mn40_as_mAP_483.pt"),
    "mn40_as(3)": urllib.parse.urljoin(model_url, "mn40_as_mAP_483(2).pt"),
    "mn40_as_no_im_pre": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_483.pt"),
    "mn40_as_no_im_pre(2)": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_483(2).pt"),
    "mn40_as_no_im_pre(3)": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_482.pt"),
    "mn40_as_ext": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_487.pt"),
    "mn40_as_ext(2)": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_486.pt"),
    "mn40_as_ext(3)": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_485.pt"),
    # varying hop size (time resolution)
    "mn10_as_hop_5": urllib.parse.urljoin(model_url, "mn10_as_hop_5_mAP_475.pt"),
    "mn10_as_hop_15": urllib.parse.urljoin(model_url, "mn10_as_hop_15_mAP_463.pt"),
    "mn10_as_hop_20": urllib.parse.urljoin(model_url, "mn10_as_hop_20_mAP_456.pt"),
    "mn10_as_hop_25": urllib.parse.urljoin(model_url, "mn10_as_hop_25_mAP_447.pt"),
    # varying n_mels (frequency resolution)
    "mn10_as_mels_40": urllib.parse.urljoin(model_url, "mn10_as_mels_40_mAP_453.pt"),
    "mn10_as_mels_64": urllib.parse.urljoin(model_url, "mn10_as_mels_64_mAP_461.pt"),
    "mn10_as_mels_256": urllib.parse.urljoin(model_url, "mn10_as_mels_256_mAP_474.pt"),
    # fully-convolutional head
    "mn10_as_fc": urllib.parse.urljoin(model_url, "mn10_as_fc_mAP_465.pt"),
    "mn10_as_fc_s2221": urllib.parse.urljoin(model_url, "mn10_as_fc_s2221_mAP_466.pt"),
    "mn10_as_fc_s2211": urllib.parse.urljoin(model_url, "mn10_as_fc_s2211_mAP_466.pt"),
}


class MN(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            in_conv_kernel: int = 3,
            in_conv_stride: int = 2,
            in_channels: int = 1,
            **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for models
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            in_conv_kernel (int): Size of kernel for first convolution
            in_conv_stride (int): Size of stride for first convolution
            in_channels (int): Number of input channels
        """
        super(MN, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        depthwise_norm_layer = norm_layer = \
            norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        kernel_sizes = [in_conv_kernel]
        strides = [in_conv_stride]

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # get squeeze excitation config
        se_cnf = kwargs.get('se_conf', None)

        # building inverted residual blocks
        # - keep track of size of frequency and time dimensions for possible application of Squeeze-and-Excitation
        # on the frequency/time dimension
        # - applying Squeeze-and-Excitation on the time dimension is not recommended as this constrains the network to
        # a particular length of the audio clip, whereas Squeeze-and-Excitation on the frequency bands is fine,
        # as the number of frequency bands is usually not changing
        f_dim, t_dim = kwargs.get('input_dims', (128, 1000))
        # take into account first conv layer
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim, idx=0)
            t_dim = cnf.out_size(t_dim, idx=1)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim  # update dimensions in block config
            layers.append(block(cnf, se_cnf, norm_layer, depthwise_norm_layer))
            kernel_sizes.append(cnf.kernel)
            strides.append(cnf.stride)

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.lastconv_output_channels = lastconv_output_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)

        # no prediction head needed - we want to use Frame-MobileNet to extract a 3D sequence
        #  i.e.: batch size x sequence length x channel dimension

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, return_fmaps: bool = False) -> Tensor:
        fmaps = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)

        # reshape: batch size x channels x frequency bands x time -> batch size x time x channels
        #  works, because frequency dimension is exactly 1
        x = x.squeeze(2).permute(0, 2, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def layerwise_lr_decay(self, lr, lr_decay):
        blocks = [name for name, _ in self.named_parameters() if name.startswith("features")]
        all_layers = blocks
        all_layers.reverse()

        parameters = []
        info = []
        prev_group_name = ".".join(all_layers[0].split('.')[:2])

        # store params & learning rates
        for idx, name in enumerate(all_layers):
            cur_group_name = ".".join(name.split('.')[:2])

            # update learning rate
            if cur_group_name != prev_group_name:
                lr *= lr_decay
            prev_group_name = cur_group_name

            # append layer parameters
            parameters += [{'params': [p for n, p in self.named_parameters() if n == name and p.requires_grad],
                            'lr': lr}]
            info.append(f"{name}: lr={lr:.6f}")

        return parameters

    def load_model(self, path, wandb_id):
        ckpt_path = os.path.join(path, wandb_id + ".ckpt")

        pretrained_weights = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        pretrained_weights = {k[10:]: v for k, v in pretrained_weights.items() if k[:10] == "net.model."}
        self.load_state_dict(pretrained_weights)

        print("Loaded model successfully. Wandb_id:", wandb_id)


def _mobilenet_v3_conf(
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: Tuple[int] = None,
        dilation_list_t_dim: Optional[List[int]] = None,
        **kwargs
):
    reduce_divider = 2 if reduced_tail else 1
    if dilation_list_t_dim is None:
        dilation_list_t_dim = [1] * 15
    if dilated:
        dilation_list_t_dim[-3:] = [2] * 3

    print("dilation_list_t_dim: ")
    print(dilation_list_t_dim)

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if strides is None:
        #            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
        f_strides = (1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2)
        t_strides = (1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

        strides = tuple(zip(f_strides, t_strides))

    # InvertedResidualConfig:
    # input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", strides[0], (1, dilation_list_t_dim[0])),  # 0
        bneck_conf(16, 3, 64, 24, False, "RE", strides[1], (1, dilation_list_t_dim[1])),  # 1 - C1
        bneck_conf(24, 3, 72, 24, False, "RE", strides[2], (1, dilation_list_t_dim[2])),  # 2
        bneck_conf(24, 5, 72, 40, True, "RE", strides[3], (1, dilation_list_t_dim[3])),  # 3 - C2
        bneck_conf(40, 5, 120, 40, True, "RE", strides[4], (1, dilation_list_t_dim[4])),  # 4
        bneck_conf(40, 5, 120, 40, True, "RE", strides[5], (1, dilation_list_t_dim[5])),  # 5
        bneck_conf(40, 3, 240, 80, False, "HS", strides[6], (1, dilation_list_t_dim[6])),  # 6 - C3
        bneck_conf(80, 3, 200, 80, False, "HS", strides[7], (1, dilation_list_t_dim[7])),  # 7
        bneck_conf(80, 3, 184, 80, False, "HS", strides[8], (1, dilation_list_t_dim[8])),  # 8
        bneck_conf(80, 3, 184, 80, False, "HS", strides[9], (1, dilation_list_t_dim[9])),  # 9
        bneck_conf(80, 3, 480, 112, True, "HS", strides[10], (1, dilation_list_t_dim[10])),  # 10
        bneck_conf(112, 3, 672, 112, True, "HS", strides[11], (1, dilation_list_t_dim[11])),  # 11
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", strides[12], (1, dilation_list_t_dim[12])),
        # 12 - C4 # dilation
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", strides[13],
                   (1, dilation_list_t_dim[13])),  # 13  # dilation
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", strides[14],
                   (1, dilation_list_t_dim[14])),  # 14  # dilation
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
        inverted_residual_setting: List[InvertedResidualConfig],
        pretrained_name: str,
        **kwargs: Any,
):
    model = MN(inverted_residual_setting, **kwargs)

    if pretrained_name in pretrained_models:
        model_url = pretrained_models.get(pretrained_name)
        state_dict = load_state_dict_from_url(model_url, model_dir=model_dir, map_location="cpu")
        if kwargs['head_type'] == "mlp":
            num_classes = state_dict['classifier.5.bias'].size(0)
        elif kwargs['head_type'] == "fully_convolutional":
            num_classes = state_dict['classifier.1.bias'].size(0)
        else:
            print("Loading weights for classifier only implemented for head types 'mlp' and 'fully_convolutional'")
            num_classes = -1
        if kwargs['num_classes'] != num_classes:
            # if the number of logits is not matching the state dict,
            # drop the corresponding pre-trained part
            pretrain_logits = state_dict['classifier.5.bias'].size(0) if kwargs['head_type'] == "mlp" \
                else state_dict['classifier.1.bias'].size(0)
            print(f"Number of classes defined: {kwargs['num_classes']}, "
                  f"but try to load pre-trained layer with logits: {pretrain_logits}\n"
                  "Dropping last layer.")
            if kwargs['head_type'] == "mlp":
                del state_dict['classifier.5.weight']
                del state_dict['classifier.5.bias']
            else:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(str(e))
            print("Loading weights pre-trained weights in a non-strict manner.")
            model.load_state_dict(state_dict, strict=False)
    elif pretrained_name:
        raise NotImplementedError(f"Model name '{pretrained_name}' unknown.")
    return model


def mobilenet_v3(pretrained_name: str = None, **kwargs: Any) \
        -> MN:
    """
    Constructs a MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>".
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return _mobilenet_v3(inverted_residual_setting, pretrained_name, **kwargs)


def get_model(pretrained_name: str = None, width_mult: float = 1.0,
              reduced_tail: bool = False, dilated: bool = False, dilation_list_t_dim=None,
              strides: Tuple[int, int, int, int] = None,
              head_type: str = "mlp", multihead_attention_heads: int = 4, input_dim_f: int = 128,
              input_dim_t: int = 1000, se_dims: str = 'c', se_agg: str = "max", se_r: int = 4):
    """
        Arguments to modify the instantiation of a MobileNetv3

        Args:
            pretrained_name (str): Specifies name of pre-trained model to load
            width_mult (float): Scales width of network
            reduced_tail (bool): Scales down network tail
            dilated (bool): Applies dilated convolution to network tail
            dilation_list_t_dim (List): List of dilation factors to apply to network tail
            strides (Tuple): Strides that are set to '2' in original implementation;
                might be changed to modify the size of receptive field and the downsampling factor in
                time and frequency dimension
            head_type (str): decides which classification head to use
            multihead_attention_heads (int): number of heads in case 'multihead_attention_heads' is used
            input_dim_f (int): number of frequency bands
            input_dim_t (int): number of time frames
            se_dims (Tuple): choose dimension to apply squeeze-excitation on, if multiple dimensions are chosen, then
                squeeze-excitation is applied concurrently and se layer outputs are fused by se_agg operation
            se_agg (str): operation to fuse output of concurrent se layers
            se_r (int): squeeze excitation bottleneck size
            se_dims (str): contains letters corresponding to dimensions 'c' - channel, 'f' - frequency, 't' - time
        """

    dim_map = {'c': 1, 'f': 2, 't': 3}
    assert len(se_dims) <= 3 and all([s in dim_map.keys() for s in se_dims]) or se_dims == 'none'
    input_dims = (input_dim_f, input_dim_t)
    if se_dims == 'none':
        se_dims = None
    else:
        se_dims = [dim_map[s] for s in se_dims]
    se_conf = dict(se_dims=se_dims, se_agg=se_agg, se_r=se_r)
    m = mobilenet_v3(pretrained_name=pretrained_name,
                     width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated,
                     dilation_list_t_dim=dilation_list_t_dim,
                     strides=strides,
                     head_type=head_type, multihead_attention_heads=multihead_attention_heads,
                     input_dims=input_dims, se_conf=se_conf
                     )
    print(m)
    return m
