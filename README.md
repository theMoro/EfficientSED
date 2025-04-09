# Efficient Pre-Trained CNNs for Sound Event Detection

In this repository, we publish pre-trained models and code for the EUSIPCO'25 paper: [**Exploring Performance-Complexity Trade-Offs in Sound Event Detection**](https://arxiv.org/abs/2503.11373).

In this paper, we propose **efficient pre-trained CNNs for Sound Event Detection (SED)** which achieve performance competitive to large state-of-the-art Audio Transformers while requiring only around 5% of the parameters. 
We create frame-wise versions of **MobileNetV3** [1] in different complexities and call them **frame-wise MobileNets (fmn)**. 
We train these models on AudioSet Weak (clip-wise labels) and subsequently on AudioSet Strong (frame-wise labels).

To optimize the performance-complexity trade-off, we investigate various sequence models to be employed on top of the frame-wise MobileNets when training on AudioSet Strong.
We show that across different complexities of the frame-wise MobileNets, the performance-complexity trade-off can be clearly improved when using the right sequence model.

We explore the following sequence models for the training on AudioSet Strong, stacked on top of the frame-wise MobileNets:
- Transformer blocks (TF) [2]
- Bidirectional Gated Recurrent Units (BiGRU) [3]
- Self-Attention layers (ATT) [2]
- Temporal Convolutional Network blocks (TCN)  [4]
- Mamba2 blocks (MAMBA) [5, 6]
- Hybrid model (HYBRID) [7] combining minGRUs [8] and self-attention

And we evaluate the **complexity** using the following three key metrics: 
- total parameter count
- multiply-accumulate operations (MACs)
- throughput (samples per second).

![Performance-Complexity trade-off of all sequence models](/images/plot_1_log.png)

**Figure 1:** Performance-Complexity trade-off of _fmn10_ trained with the various sequence models, scaled across four complexity levels. 
Colored points indicate increasing hidden dimensions (128, 256, 512, 1024) of the sequence models from left to right. For reference, 
_fmn10_ and _fmn20_ without sequence models are shown as well.

![Performance-Complexity trade-off of models scaled to different complexity levels](/images/plot_2_log.png)

**Figure 2:** Performance-Complexity trade-off of the models scaled to different complexity levels. The points on the black line, from left to right, correspond to 
the models _fmn04_, _fmn06_, _fmn10_, _fmn20_, and _fmn30_. We compare these to models incorporating transformer blocks (_fmn+TF_), HYBRID (_fmn+HYBRID_), or BiGRUs (_fmn+BiGRU_) 
with sequence models scaled proportionally to the backbone. For clarity, the BiGRU line is dashed. 
This figure shows that with the right sequence models, the performance-complexity trade-off can be improved significantly across all complexities. 


---
The code of this repository is based on [PretrainedSED](https://github.com/fschmid56/PretrainedSED), where we use Audio Transformers for Sound Event Detection, as proposed in: [**Effective Pre-Training of Audio Transformers for Sound Event Detection**](https://arxiv.org/abs/2409.09546).
These Audio Transformers are implemented in this repository as well, creating a unified framework for SED using efficient CNNs and Audio Transformers. 
The checkpoints of the Audio Transformer models are available in [PretrainedSED](https://github.com/fschmid56/PretrainedSED).

---

# Setting up the Environment

1. If needed, create a new environment with python 3.11 and activate it:

```bash
conda create -n efficient_sed python=3.11
conda activate efficient_sed
 ```

2. Install pytorch build that suits your system. For example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for cuda >= 12.1
pip3 install torch torchvision torchaudio 
```

3. Install the requirements:

 ```bash
pip3 install -r requirements.txt
 ```

4. If you intend on using the implemented Mamba code which uses the package _mamba-ssm_, run the following commands: 

```bash
pip install transformers==4.28.0
pip install triton==2.1.0
pip install mamba-ssm
```

Otherwise, you can skip this step and just comment out the import statement of _mamba-ssm_ in _models/prediction_wrapper.py_.


5. Install package for mp3 decoding:

``` bash
CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip
```

# Inference

The script `inference.py` can be used to load a pre-trained model and run sound event detection on a single audio file of arbitrary length.

```python
python inference.py --cuda --model_name="fmn10" --audio_file="test_files/752547__iscence__milan_metro_coming_in_station.wav"
```

The argument ```model_name``` specifies the model (a CNN or a transformer, e.g., fmn10 or BEATs) used for inference, and the corresponding pre-trained model checkpoint
is automatically downloaded and placed in the folder [resources](resources).

```python
python inference.py --cuda --model_name="fmn10" --seq_model_type="tf" --seq_model_dim=256 --audio_file="test_files/752547__iscence__milan_metro_coming_in_station.wav"
```

The argument ```seq_model_type``` specifies the sequence model in the dimension ```seq_model_dim``` placed on top of 
the specified model for the training on AudioSet Strong.

The argument ```audio_file``` specifies the path to a single audio file. There is one [example file](test_files/752547__iscence__milan_metro_coming_in_station.wav) included. 
More example files can be downloaded from the [GitHub release](https://github.com/fschmid56/PretrainedSED/releases/tag/v0.0.1).

# Model Checkpoints

The following is a list of checkpoints of models we trained on AudioSet Weak and AudioSet Strong. 
This table further includes the used sequence model, sequence model dimension, learning rate (_LR_) used for training the convolutional backbone on AudioSet Strong
and the learning rate used for training the sequence model (_SEQ LR_) on AudioSet Strong.

_PSDS1_ refers to the mean macro-averaged PSDS1 score of the corresponding configuration and does not refer to the PSDS1 score of the checkpoint, although they are very similar in most cases.

_Checkpoint Name_ refers to the respective names in our [GitHub release](https://github.com/theMoro/EfficientSED/releases/tag/v0.0.1). 
All model checkpoints are automatically downloaded when they are needed by running the code, or can be manually downloaded from the GitHub release.

## Checkpoints of models trained on AudioSet Weak

| Model | Checkpoint Name            |
|-------|----------------------------|
| fmn04 | fmn04_weak.pt              |
| fmn06 | fmn06_weak.pt              |
| fmn10 | fmn10_weak.pt              |
| fmn20 | fmn20_weak.pt              |
| fmn30 | fmn30_weak.pt              |

## Checkpoints of models trained on AudioSet Strong

| Model | LR   | PSDS1 | Checkpoint Name |
|-------|------|-------|-----------------|
| fmn04 | 6e-3 | 39.52 | fmn04_strong.pt |
| fmn06 | 6e-3 | 40.75 | fmn06_strong.pt |
| fmn10 | 3e-3 | 42.28 | fmn10_strong.pt |
| fmn20 | 3e-3 | 43.86 | fmn20_strong.pt |
| fmn30 | 3e-3 | 44.24 | fmn30_strong.pt |

## Checkpoints of _fmn10 + seq model_ trained on AudioSet Strong

| Model | LR   | Seq Model | Seq Model Dim | SEQ LR | PSDS1 | Checkpoint Name             |
|-------|------|-----------|---------------|--------|-------|-----------------------------|
| fmn10 | 3e-3 | TF        | 128           | 3e-3   | 42.75 | fmn10+tf-128_strong.pt      |
| fmn10 | 3e-3 | TF        | 256           | 3e-3   | 43.86 | fmn10+tf-256_strong.pt      |
| fmn10 | 3e-3 | TF        | 512           | 3e-3   | 44.11 | fmn10+tf-512_strong.pt      |
| fmn10 | 3e-3 | TF        | 1024          | 3e-3   | 44.55 | fmn10+tf-1024_strong.pt     |
| fmn10 | 3e-3 | HYBRID    | 128           | 3e-3   | 43.45 | fmn10+hybrid-128_strong.pt  |
| fmn10 | 3e-3 | HYBRID    | 256           | 8e-4   | 44.05 | fmn10+hybrid-256_strong.pt  |
| fmn10 | 3e-3 | HYBRID    | 512           | 3e-3   | 44.32 | fmn10+hybrid-512_strong.pt  |
| fmn10 | 3e-3 | HYBRID    | 1024          | 3e-3   | 44.48 | fmn10+hybrid-1024_strong.pt |
| fmn10 | 3e-3 | GRU       | 128           | 3e-3   | 43.13 | fmn10+gru-128_strong.pt     |
| fmn10 | 3e-3 | GRU       | 256           | 8e-4   | 43.53 | fmn10+gru-256_strong.pt     |
| fmn10 | 3e-3 | GRU       | 512           | 8e-4   | 43.96 | fmn10+gru-512_strong.pt     |
| fmn10 | 3e-3 | GRU       | 1024          | 8e-4   | 44.50 | fmn10+gru-1024_strong.pt    |


## Checkpoints of other _fmns + seq model_ trained on AudioSet Strong

| Model | LR   | Seq Model | Seq Model Dim | SEQ LR | PSDS1 | Checkpoint Name            |
|-------|------|-----------|---------------|--------|-------|----------------------------|
| fmn04 | 6e-3 | TF        | 104           | 3e-3   | 40.10 | fmn04+tf-104_strong.pt     |
| fmn04 | 6e-3 | HYBRID    | 104           | 3e-3   | 41.37 | fmn04+hybrid-104_strong.pt |
| fmn04 | 6e-3 | GRU       | 104           | 3e-3   | 40.81 | fmn04+gru-104_strong.pt    |
| fmn06 | 6e-3 | TF        | 160           | 3e-3   | 41.78 | fmn06+tf-160_strong.pt     |
| fmn06 | 6e-3 | HYBRID    | 160           | 3e-3   | 42.54 | fmn06+hybrid-160_strong.pt |
| fmn06 | 6e-3 | GRU       | 160           | 8e-4   | 40.37 | fmn06+gru-160_strong.pt    |
| fmn20 | 3e-3 | TF        | 512           | 3e-3   | 45.13 | fmn20+tf-512_strong.pt     |
| fmn20 | 3e-3 | HYBRID    | 512           | 3e-3   | 45.44 | fmn20+hybrid-512_strong.pt |
| fmn20 | 3e-3 | GRU       | 512           | 8e-4   | 44.94 | fmn20+gru-512_strong.pt    |
| fmn30 | 3e-3 | TF        | 768           | 3e-3   | 45.64 | fmn30+tf-768_strong.pt     |
| fmn30 | 3e-3 | HYBRID    | 768           | 3e-3   | 45.67 | fmn30+hybrid-768_strong.pt |
| fmn30 | 3e-3 | GRU       | 768           | 8e-4   | 45.77 | fmn30+gru-768_strong.pt    |

## Checkpoints trained using the Advanced Knowledge Distillation setup

| Model | LR   | Seq Model | Seq Model Dim | SEQ LR | PSDS1 | Checkpoint Name                         |
|-------|------|-----------|---------------|--------|-------|-----------------------------------------|
| fmn10 | 3e-3 | TF        | 256           | 3e-3   | 45.25 | fmn10+tf-256_advanced-kd-weak-strong.pt |


# AudioSet Strong training

## Prepare dataset

Follow the instructions in [PretrainedSED](https://github.com/fschmid56/PretrainedSED?tab=readme-ov-file#prepare-dataset) to download and prepare the dataset.

Make sure to set the environment variable ```HF_DATASETS_CACHE```. 

## Download ensemble pseudo labels

If you want to train on AudioSet Strong using Knowledge Distillation as described in the paper, follow the instructions in [PretrainedSED](https://github.com/fschmid56/PretrainedSED?tab=readme-ov-file#download-ensemble-pseudo-labels) to 
download the ensemble pseudo labels.

## Run AudioSet Strong training

You can adjust the parameters ```batch_size``` and ```accumulate_grad_batches``` directly in the command depending on the model complexity and GPU memory. 
The default values are set to 64 and 4, respectively. We used the default values for training models with a complexity similar to  _fmn10_. 
For smaller models (_fmn04_ and _fmn06_) we used a larger batch size and for larger models (_fmn20_ and _fmn30_) we used a smaller batch size.

The argument ```lr``` specifies the learning rate for the convolutional backbone and the argument ```seq_lr``` specifies the learning rate for the sequence model if used. 

The argument ```experiment_name``` specifies the name of the experiment / run in Weights & Biases. 

Example:
Train _fmn10_ with transformer blocks of the dimension 256 (_fmn10+tf:256_) following the exact same setup as described in the paper including Knowledge Distillation:  

```
python ex_audioset_strong.py --model_name=fmn10 --seq_model_type=tf --seq_model_dim=256 --pretrained=weak --max_lr=3e-3 --seq_lr=3e-3 --distillation_loss_weight=0.9 --pseudo_labels_file=<path_to_pseudo_label_file_from_Zenodo> --experiment_name=fmn10+tf:256_training
```

Example:
Train _fmn20+gru:512_ following the exact same setup as described in the paper including Knowledge Distillation:  

```
python ex_audioset_strong.py --model_name=fmn20 --seq_model_type=gru --seq_model_dim=512 --pretrained=weak --max_lr=3e-3 --seq_lr=8e-4 --batch_size=32 --accumulate_grad_batches=8 --distillation_loss_weight=0.9 --pseudo_labels_file=<path_to_pseudo_label_file_from_Zenodo> --experiment_name=fmn20+gru:512_training
```

Example: Train _fmn06+hybrid:160_ without Knowledge Distillation: 
```
python ex_audioset_strong.py --model_name=fmn06 --seq_model_type=hybrid --seq_model_dim=160 --pretrained=weak --batch_size=128 --accumulate_grad_batches=2 --max_lr=1e-4 --seq_lr=1e-4 --experiment_name=fmn06+hybrid:160_training_without_kd
```

The performance is significantly lower when training without Knowledge Distillation. 
Additionally, the learning rate needs to be adjusted when training without it. 


## Run AudioSet Strong evaluation

Evaluate the AudioSet Strong pre-trained checkpoint of _fmn10+tf:256_:

```
python ex_audioset_strong.py --model_name=fmn10 --seq_model_type=tf --seq_model_dim=256 --pretrained=strong --evaluate --experiment_name=fmn10+tf:256_evaluate_strong_model
```

If everything is set up correctly, this should give a `val/psds1_macro_averaged` of around 43.7.

## Run AudioSet Strong training using the Advanced Knowledge Distillation setup: 

The Advanced Knowledge Distillation setup refers to Section IV.D in the paper which is called _Expanding Knowledge Distillation to AudioSet Weak_.

In this setup, we leverage the top-performing transformer model on AudioSet Strong from [PretrainedSED](https://github.com/fschmid56/PretrainedSED?tab=readme-ov-file#prepare-dataset), BEATs [9], 
to generate frame-level predictions for the AudioSet Strong as well as for the AudioSet Weak training split.
The distillation loss is then computed on batches containing 50% AudioSet Weak and 50% AudioSet Strong samples.

In this setup, the model _fmn10+tf:256_ achieves an even higher PSDS1 score of 45.25 on AudioSet Strong, which is a significant improvement over the 43.86 PSDS1 score of training without the 
Advanced Knowledge Distillation setup. 

The training can be run using the following command:

```
python ex_audioset_strong_online_distillation.py --model_name=fmn10 --seq_model_type=tf --seq_model_dim=256 --pretrained=weak --experiment_name=fmn10+tf:256_online_distillation
```

**Note:** In this setup, the arguments ```weak_distillation_loss_weight``` and ```strong_distillation_loss_weight``` are both set to 0 by default, while 
the arguments ```strong_supervised_loss_weight``` and ```online_distillation_loss_weight``` are set to 0.1 and 0.9 by default, respectively.


# Fine-Tuning on Downstream Task

We demonstrate how our pre-trained models can be fine-tuned for the downstream Sound Event Detection task by using our models on [DCASE 2016 Task 2](https://dcase.community/challenge2016/task-sound-event-detection-in-synthetic-audio-results).
This task focuses on detecting office sounds and is part of the [HEAR benchmark](https://hearbenchmark.com/hear-tasks.html).

## Obtain DCASE 2016 Task 2 Dataset in HEAR format

Follow the instructions on the [HEAR website](https://hearbenchmark.com/hear-tasks.html) to download the dataset in 16 kHz sampling rate. After completing the setup, your file tree should look similar to this:
```
hear_datasets/tasks/dcase2016_task2-hear2021-full/
├── 16000
├── 48000
├── labelvocabulary.csv
├── task_metadata.json
├── test.json
├── train.json
└── valid.json
```

The ```16000``` folder contains audio files sampled at 16 kHz.

## Run Fine-Tuning

The script `ex_dcase2016task2.py` can be used to fine-tune a pre-trained model on the DCASE 2016 Task 2 dataset.

To fine-tune the full ATST-F model, pre-trained on AudioSet Strong, with a layer-wise learning rate decay of 0.95, use the following command:

```
python ex_dcase2016task2.py --task_path=hear_datasets/tasks/dcase2016_task2-hear2021-full --model_name=ATST-F --pretrained=strong --lr_decay=0.95
```

To train only the linear prediction head on top of the frozen BEATs transformer, also pre-trained on AudioSet Strong, use this command:

```
python ex_dcase2016task2.py --task_path=hear_datasets/tasks/dcase2016_task2-hear2021-full --model_name=BEATs --pretrained=strong --backbone_frozen --max_lr=2e-1 --mixup_p=0 --wavmix_p=0 --no_adamw --weight_decay=0 --n_epochs=500
```

This script can also be used to fine-tune frame-wise MobileNets on this task. 
However, we did not fine-tune the hyperparameters for training the frame-wise MobileNets on this task, so the results may not be optimal.



# References

[1] A. Howard, R. Pang, H. Adam, Q. V. Le, M. Sandler, B. Chen, W. Wang, L. Chen, M. Tan, G. Chu, V. Vasudevan, and Y. Zhu, “Searching for mobilenetv3,” in Proceedings of the International Conference on Computer Vision (ICCV), 2019.
[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 2017.
[3] K. Cho, B. van Merrienboer, C¸ . G ¨ulc¸ ehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2014, pp. 1724–1734.
[4] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. W. Senior, and K. Kavukcuoglu, “Wavenet: A generative model for raw audio,” in Speech Synthesis Workshop, 2016.
[5] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” CoRR, vol. abs/2312.00752, 2023.
[6] T. Dao and A. Gu, “Transformers are ssms: Generalized models and efficient algorithms through structured state space duality,” in Proceedings of the International Conference on Machine Learning (ICML), 2024.
[7] P. Wang, “minGRU-pytorch,” https://github.com/lucidrains/minGRU-pytorch, 2024, GitHub repository.
[8] L. Feng, F. Tung, M. O. Ahmed, Y. Bengio, and H. Hajimirsadeghi, “Were rnns all we needed?” CoRR, vol. abs/2410.01201, 2024.
[9] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, W. Che, X. Yu, and F. Wei, “BEATs: Audio pre-training with acoustic tokenizers,” in Proceedings of the International Conference on Machine Learning (ICML), 2023.