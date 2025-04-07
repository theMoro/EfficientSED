# Efficient Pre-Trained CNNs for Sound Event Detection

In this repository, we publish pre-trained models and code for the EUSIPCO'25 paper: [**Exploring Performance-Complexity Trade-Offs in Sound Event Detection**](https://arxiv.org/abs/2503.11373).

In this paper, we propose **efficient pre-trained CNNs for Sound Event Detection (SED)** which achieve performance competitive to large state-of-the-art Audio Transformers while requiring only around 5% of the parameters. 
We create frame-wise versions of **MobileNetV3** [1] in different complexities and call them **frame-wise MobileNets (fmn)**. 
We train these models on AudioSet Weak (clip-wise labels) and subsequently on AudioSet Strong (frame-wise labels).

To optimize the performance-complexity trade-off, we investigate various sequence models to be employed on top of the frame-wise MobileNets when training on AudioSet Strong.
We show that across different complexities of the frame-wise MobileNets, using appropriate sequence models clearly enhances the performance-complexity trade-off.

We explore the following sequence models for the training on AudioSet Strong, stacked on top of the frame-wise MobileNets:
- Transformer blocks (TF)
- Bidirectional Gated Recurrent Units (BiGRU)
- Self-Attention layers (ATT)
- Temporal Convolutional Network blocks (TCN)
- Mamba2 blocks (MAMBA)
- Hybrid model combining minGRUs and self-attention (HYBRID)

And we evaluate the **complexity** using the following three key metrics: 
- total parameter count
- multiply-accumulate operations (MACs)
- throughput (samples per second).

![Performance Complexity Trade Off](/images/plot_1_log.png)


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

# AudioSet Strong training

## Prepare dataset

Follow the instructions in [PretrainedSED](https://github.com/fschmid56/PretrainedSED?tab=readme-ov-file#prepare-dataset) to download and prepare the dataset.

## Download ensemble pseudo labels

If you want to train on AudioSet Strong using Knowledge Distillation as described in the paper, follow the instructions in [PretrainedSED](https://github.com/fschmid56/PretrainedSED?tab=readme-ov-file#download-ensemble-pseudo-labels) to 
download the ensemble pseudo labels.

## Run AudioSet Strong training

You can adjust the parameters ```batch_size``` and ```accumulate_grad_batches``` directly in the command depending on the model complexity and GPU memory. 
The default values are set to 64 and 4, respectively. We used the default values for training models with a complexity similar to  _fmn10_. 
For smaller models (_fmn04_ and _fmn06_) we used a larger batch size and for larger models (_fmn20_ and _fmn30_) we used a smaller batch size.

Example:
Train _fmn10_ with transformer blocks of the dimension 256 (_fmn10+tf:256_) following the exact same setup as described in the paper including Knowledge Distillation:  

```
python ex_audioset_strong.py --model_name=fmn10 --seq_model_type=tf --seq_model_dim=256 --pretrained=weak --max_lr=3e-3 --distillation_loss_weight=0.9 --pseudo_labels_file=<path_to_pseudo_label_file_from_Zenodo>
```

Example: Train _fmn20+hybrid:512_ without Knowledge Distillation: 
```
python ex_audioset_strong.py --model_name="fmn20" --seq_model_type="hybrid" --seq_model_dim=512 --pretrained="weak" --batch_size=32 --accumulate_grad_batches=8 --max_lr=1e-4
```

The training without Knowledge Distillation is not recommended, as the performance is significantly lower than with KD. Additionally, the learning rate needs to be adjusted when training without Knowledge Distillation. 


## Run AudioSet Strong evaluation

Evaluate the AudioSet Strong pre-trained checkpoint of _fmn10+tf:256_:

```
python ex_audioset_strong.py --model_name=fmn10 --seq_model_type=tf --seq_model_dim=256 --pretrained=strong --evaluate
```

If everything is set up correctly, this should give a `val/psds1_macro_averaged` of around 43.7.


# ! Repository in progress !

This repository is in progress and will be updated with the pre-trained models and the code for training the efficient models in the next few weeks.

**Update:** Repository is constantly being updated - this is not the final version.

**TODO:**
- [x] Add code for all the sequence models.
- [x] Add code for training the efficient CNNs on AudioSet Strong.
- [x] Merge the two Wrapper classes for the Audio Transformers and the Efficient CNNs.
- [ ] Add file to train the efficient models using the "Advanced KD" setup. 
- [x] Add pre-trained models.
- [x] Check inference.py for the efficient models.
- [ ] Add plots from the paper to the README file. 
- [ ] Adapt ex_dcase2016task2.py to work with the efficient models.
- [x] Update requirements.txt





# References

[1] A. Howard, R. Pang, H. Adam, Q. V. Le, M. Sandler, B. Chen, W. Wang, L. Chen, M. Tan, G. Chu, V. Vasudevan, and Y. Zhu, “Searching for mobilenetv3,” in Proceedings of the International Conference on Computer Vision (ICCV), 2019.