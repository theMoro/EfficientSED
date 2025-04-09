from models.transformers.frame_passt.preprocess import AugmentMelSTFT
from models.transformers.transformer_wrapper import BaseModelWrapper
from models.efficient_cnns.fmn.model import get_model


class FrameMNWrapper(BaseModelWrapper):
    def __init__(self, width_mult=1.0) -> None:
        super().__init__()
        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=16_000,
            win_length=400,
            hopsize=160,
            n_fft=512,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
            fast_norm=True,
            preamp=True,
            padding="center",
            periodic_window=False,
        )

        self.fmn = get_model(
            width_mult=width_mult
        )

    def layerwise_lr_decay(self, lr, lr_decay):
        return self.fmn.layerwise_lr_decay(lr, lr_decay)

    def mel_forward(self, x):
        return self.mel(x)

    def forward(self, x):
        return self.fmn(x)

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.named_parameters():
            if any(['cls_token' in k,
                    'pos_embed' in k,
                    'norm_stats' in k,
                    'patch_embed' in k]):
                pt_params[0].append(p)
            elif 'features.0.' in k:
                pt_params[0].append(p)
            elif 'features.1.' in k:
                pt_params[1].append(p)
            elif 'features.2.' in k:
                pt_params[2].append(p)
            elif 'features.3.' in k:
                pt_params[3].append(p)
            elif 'features.4.' in k:
                pt_params[4].append(p)
            elif 'features.5.' in k:
                pt_params[5].append(p)
            elif 'features.6.' in k:
                pt_params[6].append(p)
            elif 'features.7.' in k:
                pt_params[7].append(p)
            elif 'features.8.' in k:
                pt_params[8].append(p)
            elif 'features.9.' in k:
                pt_params[9].append(p)
            elif 'features.10.' in k:
                pt_params[10].append(p)
            elif 'features.11.' in k:
                pt_params[11].append(p)
            elif 'features.12.' in k:
                pt_params[12].append(p)
            elif 'features.13.' in k:
                pt_params[13].append(p)
            elif 'features.14.' in k:
                pt_params[14].append(p)
            elif 'features.15.' in k:
                pt_params[15].append(p)
            elif 'features.16.' in k:
                pt_params[16].append(p)
            else:
                raise ValueError(f"Check separate params for fmn! Unknown key: {k}")
        return list(reversed(pt_params))
