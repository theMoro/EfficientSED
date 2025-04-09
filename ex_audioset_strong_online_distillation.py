import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities import CombinedLoader
from torch import autocast
# from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import wandb
import transformers
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import sed_scores_eval

from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from helpers.mixup import get_mixup_coefficients, apply_mixup
from models.transformers.atstframe.ATSTF_wrapper import ATSTWrapper
from models.transformers.beats.BEATs_wrapper import BEATsWrapper
from models.transformers.frame_passt.fpasst_wrapper import FPaSSTWrapper
from models.transformers.m2d.M2D_wrapper import M2DWrapper
from models.transformers.asit.ASIT_wrapper import ASiTWrapper
from models.prediction_wrapper import PredictionsWrapper
from helpers.augment import frame_shift, time_mask, mixup, filter_augmentation, mixstyle, RandomResizeCrop
from helpers.utils import worker_init_fn
from data_util.audioset_strong import get_training_dataset, get_eval_dataset
from data_util import audioset
from data_util.audioset_strong import get_temporal_count_balanced_sample_weights, get_uniform_sample_weights, \
    get_weighted_sampler
from data_util.audioset_classes import as_strong_train_classes, as_strong_eval_classes
from models.efficient_cnns.fmn.fmn_wrapper import FrameMNWrapper
from models.efficient_cnns.fmn.utils import NAME_TO_WIDTH


class PLModule(pl.LightningModule):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder

        if config.model_name in ["BEATs", "ATST-F", "fpasst", "M2D", "ASIT"]:
            if config.pretrained == "scratch":
                checkpoint = None
            elif config.pretrained == "ssl":
                checkpoint = "ssl"
            elif config.pretrained == "weak":
                checkpoint = "weak"
            elif config.pretrained == "strong":
                checkpoint = "strong_1"
            else:
                raise ValueError(f"Unknown pretrained checkpoint: {config.pretrained}")
        elif config.model_name.startswith("fmn"):
            if config.pretrained in ["weak", "strong", "advanced-kd-weak-strong"]:
                checkpoint = config.pretrained
            elif config.pretrained == "scratch":
                checkpoint = None
            else: raise ValueError(f"Invalid configuration for fmn model: config.pretrained={config.pretrained}")
        else:
            raise ValueError(f"Invalid model_name!")

        # load transformer model
        # as PretrainedSED provides checkpoints for pretrained transformers WITHOUT a sequence model on top of it,
        # we load the pretrained model transformer, throw away its head and initialize the sequence model and the new head (transformer -> seq_model -> head)
        # --> only the transformer model is pretrained (if checkpoint is not None), the sequence model and the head are trained from scratch
        if config.model_name == "BEATs":
            beats = BEATsWrapper()
            model = PredictionsWrapper(beats, checkpoint=f"BEATs_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type, seq_model_dim=2048)
        elif config.model_name == "ATST-F":
            atst = ATSTWrapper()
            model = PredictionsWrapper(atst, checkpoint=f"ATST-F_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type, seq_model_dim=2048)
        elif config.model_name == "fpasst":
            fpasst = FPaSSTWrapper()
            model = PredictionsWrapper(fpasst, checkpoint=f"fpasst_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type, seq_model_dim=2048)
        elif config.model_name == "M2D":
            m2d = M2DWrapper()
            model = PredictionsWrapper(m2d, checkpoint=f"M2D_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type,
                                       embed_dim=m2d.m2d.cfg.feature_d,
                                       seq_model_dim=2048)
        elif config.model_name == "ASIT":
            asit = ASiTWrapper()
            model = PredictionsWrapper(asit, checkpoint=f"ASIT_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type, seq_model_dim=2048)
        # or load CNN model
        # If config.pretrained == "strong", we load models (with or without sequence model on top) pretrained on AudioSet Strong --> sequence model is pretrained
        # If config.pretrained == "weak", we load models WITHOUT a sequence model pretrained on AudioSet Weak --> sequence model is trained from scratch
        elif config.model_name.startswith("fmn"):
            width = NAME_TO_WIDTH(config.model_name)
            fmn = FrameMNWrapper(width)
            embed_dim = fmn.state_dict()['fmn.features.16.1.bias'].shape[0]  # TODO: check if it works

            # build checkpoint name
            checkpoint_name = None

            if checkpoint:
                seq_model_name = ""
                if config.seq_model_type and checkpoint in ["strong", "advanced-kd-weak-strong"]:  # sequence models are only used for training on AS Strong
                    seq_model_name = f"+{config.seq_model_type}-{config.seq_model_dim}"

                checkpoint_name = f"{config.model_name}{seq_model_name}_{checkpoint}"

            model = PredictionsWrapper(
                            fmn,
                            checkpoint=checkpoint_name,
                            seq_model_type=config.seq_model_type,
                            seq_model_dim=config.seq_model_dim,
                            embed_dim=embed_dim
                            )
        else:
            raise NotImplementedError(f"Model {config.model_name} not (yet) implemented")

        self.model = model

        # prepare ingredients for knowledge distillation
        if config.online_distillation_loss_weight > 0:
            # load top single AudioSet Strong model (= BEATs)
            beats = BEATsWrapper()
            self.online_teacher = PredictionsWrapper(beats, checkpoint=f"BEATs_strong_1" if checkpoint else None,
                                       seq_model_type=None)

            for param in self.online_teacher.parameters():
                param.detach_()
        else:
            self.online_teacher = None

        self.weak_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.weak_distillation_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.strong_loss = nn.BCEWithLogitsLoss()

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self.val_durations_df = pd.read_csv(f"resources/eval_durations.csv",
                                            sep=",", header=None, names=["filename", "duration"])
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []

    def forward(self, batch):
        x = batch["audio"]
        mel = self.model.mel_forward(x)
        y_strong, y_weak = self.model(mel)
        return y_strong, y_weak

    def forward_weak(self, batch):
        """
        Computes a complete forward pass with augmentations for a weakly labeled batch.
        Takes a batch (dictionary) as input and returns the same batch with additional keys.
        """
        x_audio = batch["audio"]
        y = batch["target"]

        # sanity check
        if torch.isnan(x_audio).any() or torch.isinf(x_audio).any():
            print("Input data contains NaN or infinite values.")

        # compute audio-visual features
        x_mel = self.model.mel_forward(x_audio)

        if self.online_teacher is not None:
            if hasattr(self.online_teacher, "eval"):
                self.online_teacher.eval()
            x_mel_online_teacher = self.online_teacher.mel_forward(x_audio)

        # augmentations
        mixup_config = None
        if self.training:
            # MixUp inputs & targets
            batch_size = len(y)
            if self.config.as_weak_mixup_p > random.random():
                mixup_config = get_mixup_coefficients(batch_size, self.config.as_weak_mixup_alpha)
                mixup_config = (mixup_config[0].to(x_mel.device), mixup_config[1].to(x_mel.device))
                permutation_indices, lam = mixup_config
                x_mel = apply_mixup(x_mel, permutation_indices, lam)
                if self.online_teacher is not None:
                    x_mel_online_teacher = apply_mixup(x_mel_online_teacher, permutation_indices, lam)
                y = y * lam.reshape(batch_size, 1) + y[permutation_indices] * (1. - lam.reshape(batch_size, 1))

            if self.config.as_weak_freq_warp_p > random.random():
                # Save the current random state
                rng_state = torch.random.get_rng_state()

                # Generate a seed (or fix a specific one for consistency across runs)
                seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

                x_mel = x_mel.squeeze(1)
                x_mel = self.freq_warp(x_mel, seed=seed)
                x_mel = x_mel.unsqueeze(1)

                if self.online_teacher is not None:
                    x_mel_online_teacher = x_mel_online_teacher.squeeze(1)
                    x_mel_online_teacher = self.freq_warp(x_mel_online_teacher, seed=seed)
                    x_mel_online_teacher = x_mel_online_teacher.unsqueeze(1)

                # Restore the random state
                torch.random.set_rng_state(rng_state)

        # forward through network; use weak head
        y_hat_strong, y_hat = self.model(x_mel)

        # store things in batch for loss computation
        batch['y'] = y
        batch['y_hat'] = y_hat
        batch['y_hat_strong'] = y_hat_strong
        batch['mixup_config'] = mixup_config

        if self.online_teacher is not None:
            self.online_teacher.eval()
            with torch.no_grad():
                y_hat_strong_online_teacher, _ = self.online_teacher(x_mel_online_teacher)
            batch['y_hat_strong_online_teacher'] = y_hat_strong_online_teacher

        return batch

    def forward_strong(self, batch):
        """
        Computes a complete forward pass with augmentations for a weakly labeled batch.
        Takes a batch (dictionary) as input and returns the same batch with additional keys.
        """
        x = batch["audio"]
        labels = batch['strong']
        if 'pseudo_strong' in batch:
            pseudo_labels = batch['pseudo_strong']
        else:
            # create dummy pseudo labels
            pseudo_labels = torch.zeros_like(labels)

        features = self.model.mel_forward(x)

        if self.online_teacher is not None:
            if hasattr(self.online_teacher, "eval"):
                self.online_teacher.eval()
            x_mel_online_teacher = self.online_teacher.mel_forward(x)
        else:
            x_mel_online_teacher = torch.zeros_like(features)

        # augmentations
        if self.training:
            # time rolling
            if self.config.frame_shift_range > 0:
                features, x_mel_online_teacher, labels, pseudo_labels = frame_shift(
                    features,
                    labels,
                    embeddings=x_mel_online_teacher,
                    pseudo_labels=pseudo_labels,
                    net_pooling=self.encoder.net_pooling,
                    shift_range=self.config.frame_shift_range
                )

            # mixup
            if self.config.mixup_p > random.random():
                features, x_mel_online_teacher, labels, pseudo_labels = mixup(
                    features,
                    embeddings=x_mel_online_teacher,
                    targets=labels,
                    pseudo_strong=pseudo_labels,
                    alpha=0.3
                )

            # mixstyle
            if self.config.mixstyle_p > random.random():
                features = mixstyle(
                    features,
                    embeddings=x_mel_online_teacher,
                    alpha=0.3
                )

            # time masking
            if self.config.max_time_mask_size > 0:
                features, x_mel_online_teacher, labels, pseudo_labels = time_mask(
                    features,
                    labels,
                    embeddings=x_mel_online_teacher,
                    pseudo_labels=pseudo_labels,
                    net_pooling=self.encoder.net_pooling,
                    max_mask_ratio=self.config.max_time_mask_size
                )

            # frequency masking
            if self.config.filter_augment_p > random.random():
                # Save the current random state
                rng_state = torch.random.get_rng_state()

                # Generate a seed (or fix a specific one for consistency across runs)
                seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

                features, _ = filter_augmentation(
                    features,
                    seed=seed
                )

                if self.online_teacher is not None:
                    x_mel_online_teacher, _ = filter_augmentation(
                        x_mel_online_teacher,
                        seed=seed
                    )

                # Restore the random state
                torch.random.set_rng_state(rng_state)

            if self.config.freq_warp_p > random.random():
                # Save the current random state
                rng_state = torch.random.get_rng_state()

                # Generate a seed (or fix a specific one for consistency across runs)
                seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

                features = features.squeeze(1)
                features = self.freq_warp(features, seed=seed)
                features = features.unsqueeze(1)

                if self.online_teacher is not None:
                    x_mel_online_teacher = x_mel_online_teacher.squeeze(1)
                    x_mel_online_teacher = self.freq_warp(x_mel_online_teacher, seed=seed)
                    x_mel_online_teacher = x_mel_online_teacher.unsqueeze(1)

                # Restore the random state
                torch.random.set_rng_state(rng_state)

            batch['pseudo_strong'] = pseudo_labels

        # forward through network; use strong head
        y_hat_strong, y_hat = self.model(features)

        # store things in batch for loss computation
        batch['y_hat'] = y_hat
        batch['y_hat_strong'] = y_hat_strong
        batch['y_strong'] = labels

        if self.online_teacher is not None:
            self.online_teacher.eval()
            with torch.no_grad():
                y_hat_strong_online_teacher, _ = self.online_teacher(x_mel_online_teacher)
            batch['y_hat_strong_online_teacher'] = y_hat_strong_online_teacher

        return batch

    def get_optimizer(
            self, lr, seq_lr, lr_decay=0.0, adamw=False, weight_decay=0.01, betas=(0.9, 0.999)
    ):
        # we split the parameters into two groups, one for the backbone model (transformer or CNN) and one for the sequence model
        # we also split each of them into <=1 dimensional and >=2 dimensional parameters, so we can only
        # apply weight decay to the >=2 dimensional parameters, thus excluding biases and batch norms, an idea from NanoGPT
        backbone_model_params_leq1D = []
        backbone_model_params_geq2D = []
        seq_model_params_leq1D = []
        seq_model_params_geq2D = []
        n_scaled_layers = 0

        if seq_lr is None:
            seq_lr = lr

        if lr_decay != 0.0:
            param_groups = self.model.model.layerwise_lr_decay(lr, lr_decay)
            for name, param in self.model.named_parameters():
                if not name.startswith('model.'):
                    # sequence model + head
                    param_groups.append({'params': param, 'lr': seq_lr})
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name.startswith('model.'):
                        if param.ndimension() >= 2:
                            backbone_model_params_geq2D.append(param)
                        elif param.ndimension() <= 1:
                            backbone_model_params_leq1D.append(param)
                        n_scaled_layers += 1
                    else:  # sequence model + head
                        if param.ndimension() >= 2:
                            seq_model_params_geq2D.append(param)
                        elif param.ndimension() <= 1:
                            seq_model_params_leq1D.append(param)

            param_groups = [
                {'params': backbone_model_params_leq1D, 'lr': lr},  # backbone model
                {'params': backbone_model_params_geq2D, 'lr': lr, 'weight_decay': weight_decay},  # backbone model
                {'params': seq_model_params_leq1D, 'lr': seq_lr},  # sequence model
                {'params': seq_model_params_geq2D, 'lr': seq_lr, 'weight_decay': weight_decay}  # sequence model
            ]

        if weight_decay > 0:
            assert adamw
        assert len(param_groups) > 0
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        return torch.optim.Adam(param_groups, lr=lr, betas=betas)

    def get_lr_scheduler(
            self,
            optimizer,
            num_training_steps,
            schedule_mode="cos",
            gamma: float = 0.999996,
            num_warmup_steps=20000,
            lr_end=2e-7,
    ):
        if schedule_mode in {"exp"}:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if schedule_mode in {"cosine", "cos"}:
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if schedule_mode in {"linear"}:
            print("Linear schedule!")
            return transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0,
                lr_end=lr_end,
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = self.get_optimizer(self.config.max_lr, self.config.seq_lr, self.config.lr_decay, adamw=self.config.adamw,
                                       weight_decay=self.config.weight_decay)

        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = self.get_lr_scheduler(optimizer, num_training_steps,
                                          schedule_mode=self.config.schedule_mode,
                                          lr_end=self.config.lr_end)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, batch, batch_idx):
        """
        :param batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """

        if "strong" in batch and "weak" in batch:
            weak_batch = batch["weak"]
            strong_batch = batch["strong"]
        elif "strong" in batch:
            weak_batch = None
            strong_batch = batch["strong"]
        else:
            weak_batch = batch["weak"]
            strong_batch = None

        if weak_batch is not None:
            # forward the weak batch supervised
            weak_batch = self.forward_weak(weak_batch)

            # compute the weak supervised loss
            with autocast(enabled=True, device_type="cuda"):
                weak_batch['y_hat_strong'] = weak_batch['y_hat_strong'].float()

                y_hat = weak_batch['y_hat'].float()
                y = weak_batch['y'].float()

                if self.config.weak_supervised_loss_weight > 0:
                    weak_supervised_loss = self.weak_loss(y_hat, y).mean()
                else:
                    weak_supervised_loss = torch.tensor(0., device=weak_batch["audio"].device,
                                                        dtype=weak_batch["audio"].dtype)

                # compute distillation loss
                if self.config.weak_distillation_loss_weight > 0:
                    weak_distillation_loss = self.weak_distillation_loss(
                        weak_batch,
                        self.weak_distillation_loss,
                        weak_batch["mixup_config"]
                    )
                else:
                    weak_distillation_loss = torch.tensor(0., device=weak_batch["audio"].device,
                                                          dtype=weak_batch["audio"].dtype)

                if self.config.online_distillation_loss_weight > 0:
                    weak_batch['y_hat_strong_online_teacher'] = weak_batch['y_hat_strong_online_teacher'].float()
                    weak_online_teacher_loss = self.strong_loss(weak_batch["y_hat_strong"],
                                                                   torch.sigmoid(weak_batch['y_hat_strong_online_teacher']))
                else:
                    weak_online_teacher_loss = torch.tensor(0., device=weak_batch["audio"].device,
                                                            dtype=weak_batch["audio"].dtype)

        else:
            weak_supervised_loss = torch.tensor(0., device=strong_batch["audio"].device,
                                                dtype=strong_batch["audio"].dtype)
            weak_distillation_loss = torch.tensor(0., device=strong_batch["audio"].device,
                                                  dtype=strong_batch["audio"].dtype)
            weak_online_teacher_loss = torch.tensor(0., device=strong_batch["audio"].device,
                                                    dtype=strong_batch["audio"].dtype)

        if strong_batch is not None:
            strong_batch = self.forward_strong(strong_batch)

            with autocast(enabled=True, device_type="cuda"):
                # compute the weak supervised loss
                y_hat_strong = strong_batch["y_hat_strong"].float()
                y_strong = strong_batch["y_strong"].float()

                strong_supervised_loss = self.strong_loss(y_hat_strong, y_strong)
                # return 0 if weight is zero
                if self.config.strong_distillation_loss_weight <= 0:
                    strong_distillation_loss = torch.tensor(0., device=y_hat_strong.device, dtype=y_hat_strong.dtype)
                else:
                    pseudo_strong = strong_batch["pseudo_strong"].float()
                    strong_distillation_loss = self.strong_loss(y_hat_strong, pseudo_strong)

                if self.config.online_distillation_loss_weight > 0:
                    strong_batch['y_hat_strong_online_teacher'] = strong_batch['y_hat_strong_online_teacher'].float()
                    strong_online_teacher_loss = self.strong_loss(strong_batch["y_hat_strong"],
                                                                     torch.sigmoid(
                                                                         strong_batch['y_hat_strong_online_teacher']))
                else:
                    strong_online_teacher_loss = torch.tensor(0., device=strong_batch["audio"].device,
                                                              dtype=strong_batch["audio"].dtype)
        else:
            strong_supervised_loss = torch.tensor(0., device=weak_batch["audio"].device,
                                                  dtype=weak_batch["audio"].dtype)
            strong_distillation_loss = torch.tensor(0., device=weak_batch["audio"].device,
                                                    dtype=weak_batch["audio"].dtype)
            strong_online_teacher_loss = torch.tensor(0., device=weak_batch["audio"].device,
                                                      dtype=weak_batch["audio"].dtype)


        loss = (
                self.config.weak_supervised_loss_weight * weak_supervised_loss +
                self.config.weak_distillation_loss_weight * weak_distillation_loss +
                self.config.strong_supervised_loss_weight * strong_supervised_loss +
                self.config.strong_distillation_loss_weight * strong_distillation_loss +
                self.config.online_distillation_loss_weight * (
                        self.config.online_distillation_strong_ratio * strong_online_teacher_loss +
                        (1 - self.config.online_distillation_strong_ratio) * weak_online_teacher_loss
                )
        )

        # logging
        self.log('epoch', self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_optimizer_{i}', param_group['lr'])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)
        self.log("train/weak_supervised_loss", weak_supervised_loss.detach().cpu())
        self.log("train/weak_distillation_loss", weak_distillation_loss.detach().cpu())
        self.log("train/weak_online_teacher_loss", weak_online_teacher_loss.detach().cpu())
        self.log("train/strong_supervised_loss", strong_supervised_loss.detach().cpu())
        self.log("train/strong_distillation_loss", strong_distillation_loss.detach().cpu())
        self.log("train/strong_online_teacher_loss", strong_online_teacher_loss.detach().cpu())

        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):

        assert dataloader_idx in [0, 1], "Only weak and strong evaluation are supported."

        # distinguish between weak and strong eval sets
        if 'gt_string' not in val_batch:
            # weak evaluation
            batch = self.forward_weak(val_batch)
            y_hat = batch['y_hat']
            y = batch['y']
            nan_mask = torch.isnan(y_hat)
            y_hat = torch.nan_to_num(y_hat, nan=0.0)

            assert not torch.isnan(y_hat).any(), f"y_hat contains NaN values."
            assert not torch.isnan(y).any(), f"y contains NaN values."

            with autocast(enabled=True, device_type="cuda"):
                y_hat = y_hat.float()
                y = y.float()

                samples_loss = self.weak_loss(y_hat, y)

            y_hat = torch.sigmoid(y_hat.detach())
            results = {
                "loss": samples_loss.detach(),
                "y_hat": y_hat.detach(),
                "target": y.detach(),
                "nan_mask": nan_mask.detach()
            }
            results = {k: v.cpu() for k, v in results.items()}
            self.val_step_outputs_weak.append(results)

        else:
            # eval strong
            # parse ground truth
            for f, gt_string in zip(val_batch["filename"], val_batch["gt_string"]):
                if f in self.val_ground_truth:
                    continue
                else:
                    f_id = f.split(".")[0]
                    events = [e.split(";;") for e in gt_string.split("++")]
                    self.val_ground_truth[f_id] = [(float(e[0]), float(e[1]), e[2]) for e in events]
                    self.val_duration[f_id] = self.val_durations_df[self.val_durations_df["filename"] == f_id]["duration"].values[0]

            val_batch = self.forward_strong(val_batch)
            y_hat_strong = val_batch['y_hat_strong']
            y_strong = val_batch['strong']

            loss = self.strong_loss(y_hat_strong, y_strong)
            self.val_loss.append(loss.item().cpu())

            _, scores_postprocessed, _ = batched_decode_preds(
                y_hat_strong.float(),
                val_batch['filename'],
                self.encoder,
                median_filter=self.config.median_window
            )

            self.val_predictions_strong.update(
                scores_postprocessed
            )

    def on_validation_epoch_end(self):
        gt_unique_events = set([e[2] for f, events in self.val_ground_truth.items() for e in events])
        train_unique_events = set(self.encoder.labels)
        # evaluate on all classes that are in both train and test sets (407 classes)
        class_intersection = gt_unique_events.intersection(train_unique_events)

        assert len(class_intersection) == len(set(as_strong_train_classes).intersection(as_strong_eval_classes)) == 407, \
            f"Intersection unique events. Expected: {len(set(as_strong_train_classes).intersection(as_strong_eval_classes))}," \
            f" Actual: {len(class_intersection)}"

        # filter ground truth according to class_intersection
        val_ground_truth = {fid: [event for event in self.val_ground_truth[fid] if event[2] in class_intersection]
                            for fid in self.val_ground_truth}
        # drop audios without events - aligned with DESED evaluation procedure
        val_ground_truth = {fid: events for fid, events in val_ground_truth.items() if len(events) > 0}
        # keep only corresponding audio durations
        audio_durations = {
            fid: self.val_duration[fid] for fid in val_ground_truth.keys()
        }

        # filter files in predictions
        as_strong_preds = {
            fid: self.val_predictions_strong[fid] for fid in val_ground_truth.keys()
        }
        # filter classes in predictions
        unused_classes = list(set(self.encoder.labels).difference(class_intersection))
        for f, df in as_strong_preds.items():
            df.drop(columns=list(unused_classes), axis=1, inplace=True)

        segment_based_pauroc = sed_scores_eval.segment_based.auroc(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1
        )

        psds1 = sed_scores_eval.intersection_based.psds(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            num_jobs=1
        )

        # "val/psds1_macro_averaged" is psds1 without penalization for performance
        #  variations across classes
        logs = {"val/loss": torch.as_tensor(self.val_loss).mean().cuda(),
                "val/psds1": psds1[0],
                "val/psds1_macro_averaged": np.array([v for k, v in psds1[1].items()]).mean(),
                "val/pauroc": segment_based_pauroc[0]['mean'],
                }

        self.log_dict(logs, sync_dist=False)
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []


def train(config):
    # Train Models on temporally-strong portion of AudioSet.

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="EfficientSED",
        notes="Training efficient models for Sound Event Detection on AudioSet Strong.",
        tags=["AudioSet Strong", "Sound Event Detection", "Pseudo Labels", "Knowledge Distillation"],
        config=config,
        name=config.experiment_name
    )

    # encoder manages encoding and decoding of model predictions
    encoder = ManyHotEncoder(as_strong_train_classes)

    # AudioSet Weak
    as_weak_train_set = audioset.get_training_dataset()

    if config.as_weak_sampling_strategy == 'uniform':
        as_weak_sample_weights = audioset.get_uniform_sample_weights(as_weak_train_set)
    elif config.as_weak_sampling_strategy == 'count':
        save_folder = 'resources'  # '/opt/scratch/as_strong/'  # TODO: change it to resources, fix this problem!
        as_weak_sample_weights = audioset.get_ft_cls_balanced_sample_weights(as_weak_train_set, save_folder=save_folder)
    else:
        raise ValueError(f"Unknown as_weak_sampling_strategy: {config.as_weak_sampling_strategy}")

    as_weak_train_sampler = audioset.get_weighted_sampler(as_weak_sample_weights)

    as_weak_train_dl = DataLoader(dataset=as_weak_train_set,
                                      sampler=as_weak_train_sampler,
                                      worker_init_fn=worker_init_fn,
                                      num_workers=config.num_workers,
                                      batch_size=config.batch_size,
                                      shuffle=None)

    as_weak_val_set = audioset.get_validation_dataset()
    as_weak_val_sampler = audioset.ValidationDistributedSampler(as_weak_val_set)
    as_weak_val_dl = DataLoader(dataset=as_weak_val_set,
                                    sampler=as_weak_val_sampler,
                                    worker_init_fn=worker_init_fn,
                                    num_workers=config.num_workers,
                                    batch_size=config.batch_size
                                    )


    # AudioSet Strong
    as_strong_train_set = get_training_dataset(encoder, wavmix_p=config.wavmix_p)
    eval_set = get_eval_dataset(encoder)

    if config.use_balanced_sampler:
        as_strong_sample_weights = get_temporal_count_balanced_sample_weights(as_strong_train_set, save_folder="resources")
    else:
        as_strong_sample_weights = get_uniform_sample_weights(as_strong_train_set)

    train_sampler = get_weighted_sampler(as_strong_sample_weights, epoch_len=config.epoch_len)

    # AudioSet Strong train dataloader
    as_strong_train_dl = DataLoader(dataset=as_strong_train_set,
                          sampler=train_sampler,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=None)

    # eval dataloader
    eval_dl = DataLoader(dataset=eval_set,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # combine training dataloaders
    if config.strong_supervised_loss_weight == 0:
        iterables_train = {'weak': as_weak_train_dl}
    elif config.weak_supervised_loss_weight == 0 and config.weak_distillation_loss_weight == 0 and \
            config.online_distillation_loss_weight == 0:

        iterables_train = {'strong': as_strong_train_dl}
    else:
        iterables_train = {'weak': as_weak_train_dl, 'strong': as_strong_train_dl}

    combined_train_loader = CombinedLoader(iterables_train, mode='min_size')


    # combine validation dataloaders
    if config.strong_supervised_loss_weight == 0:
        combined_val_loader = as_weak_val_dl
    elif config.weak_supervised_loss_weight == 0 and config.weak_distillation_loss_weight == 0:
        combined_val_loader = eval_dl
    else:
        combined_val_loader = [eval_dl, as_weak_val_dl]

    # create pytorch lightening module
    pl_module = PLModule(config, encoder)

    # create the pytorch lightning trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         accumulate_grad_batches=config.accumulate_grad_batches
                         )

    # start training and validation for the specified number of epochs
    trainer.fit(
        pl_module,
        train_dataloaders=combined_train_loader,
        val_dataloaders=combined_val_loader
    )

    wandb.finish()


def evaluate(config):
    # only evaluation of pre-trained models
    # encoder manages encoding and decoding of model predictions
    encoder = ManyHotEncoder(as_strong_train_classes)
    eval_set = get_eval_dataset(encoder)

    # eval dataloader
    eval_dl = DataLoader(dataset=eval_set,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config, encoder)

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         accumulate_grad_batches=config.accumulate_grad_batches
                         )

    # start evaluation
    trainer.validate(pl_module, eval_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet_Strong")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)

    # model
    parser.add_argument('--model_name', type=str,
                        choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"] + \
                                [f"fmn{width}" for width in ["04", "06", "10", "20", "30"]],
                        default="fmn10")  # used also for training
    # "scratch" = no pretraining
    # "ssl" = SSL pre-trained
    # "weak" = AudioSet Weak pre-trained
    # "strong" = AudioSet Strong pre-trained
    # "advanced-kd-weak-strong" = pre-trained using advanced KD on AudioSet Weak and Strong as described in Section IV.D in the paper
    parser.add_argument('--pretrained', type=str, choices=["scratch", "ssl", "weak", "strong", "advanced-kd-weak-strong"],
                        default="weak")
    parser.add_argument('--seq_model_type', type=str, choices=[None, "gru", "attn", "tf", "mamba", "tcn", "hybrid"],
                        default=None)
    parser.add_argument('--seq_model_dim', type=int, default=256)

    # training
    parser.add_argument('--n_epochs', type=int, default=240)
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False)
    parser.add_argument('--epoch_len', type=int, default=100_000)
    parser.add_argument('--median_window', type=int, default=9)

    # augmentations for AudioSet Strong
    parser.add_argument('--wavmix_p', type=float, default=0.5)
    parser.add_argument('--freq_warp_p', type=float, default=0.0)
    parser.add_argument('--filter_augment_p', type=float, default=0.0)
    parser.add_argument('--frame_shift_range', type=float, default=0.125)  # in seconds
    parser.add_argument('--mixup_p', type=float, default=0.3)
    parser.add_argument('--mixstyle_p', type=float, default=0.0)
    parser.add_argument('--max_time_mask_size', type=float, default=0.0)

    # augmentations for AudioSet Weak
    parser.add_argument('--as_weak_mixup_alpha', type=float, default=0.3)
    parser.add_argument('--as_weak_mixup_p', type=float, default=0.3)
    parser.add_argument('--as_weak_freq_warp_p', type=float, default=0.0)


    # optimizer
    parser.add_argument('--adamw', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # lr schedule
    parser.add_argument('--schedule_mode', type=str, default="cos")

    ## max_lr is for the convolutional backbone, seq_lr is for the sequence model
    parser.add_argument('--max_lr', type=float, default=8e-4)
    parser.add_argument('--seq_lr', type=float, default=None)

    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_end', type=float, default=2e-7)
    parser.add_argument('--warmup_steps', type=int, default=5000)

    parser.add_argument('--as_weak_sampling_strategy', type=str, default='count', choices=['uniform', 'count'])

    # knowledge distillation
    parser.add_argument('--weak_supervised_loss_weight', type=float, default=0.0)
    parser.add_argument('--weak_distillation_loss_weight', type=float, default=0.0)
    parser.add_argument('--strong_supervised_loss_weight', type=float, default=0.1)
    parser.add_argument('--strong_distillation_loss_weight', type=float, default=0.0)
    parser.add_argument('--online_distillation_loss_weight', type=float, default=0.9)

    parser.add_argument('--online_distillation_strong_ratio', type=float, default=0.5)

    args = parser.parse_args()

    if args.seq_model_type == "mamba":
        # Mamba requires 16-mixed precision for stable training
        warnings.warn("Using Mamba requires 16-mixed precision for stable training. Setting precision to 16-mixed.")
        args.precision = "16-mixed"

    train(args)
