# Efficient Pre-Trained CNNs for Sound Event Detection

In this repository, we publish pre-trained models and code for the EUSIPCO'25 paper: [**Exploring Performance-Complexity Trade-Offs in Sound Event Detection**](https://arxiv.org/abs/2503.11373).

In this paper, we propose **efficient pre-trained CNNs for Sound Event Detection (SED)** which achieve performance competitive to the large state-of-the-art Audio Transformers while requiring only around 5% of the parameters. 
We train frame-wise versions of **MobileNetV3** [1] first on AudioSet Weak (clip-wise labels) and subsequently on AudioSet Strong (frame-wise labels). 
To optimize the performance-complexity trade-off, we investigate various sequence models to be employed on top of the CNNs.
We show that using appropriate sequence models clearly enhances the performance-complexity trade-off, and this improvement holds across different complexities of the models.

We explore the following sequence models, stacked on top of the pre-trained MobileNetV3:
- Transformer Blocks (TF)
- BiGRU
- Attention (ATT)
- Temporal Convolutional Networks (TCN)
- Mamba (MAMBA)
- Hybrid model combining minGRUs and self-attention (HYBRID) 


We evaluate **complexity** using three key metrics: 
- total parameter count
- multiply-accumulate operations (MACs)
- throughput (samples per second).

---
The code of this repository is based on [PretrainedSED](https://github.com/fschmid56/PretrainedSED), where we use Audio Transformers for Sound Event Detection, as proposed in: [**Effective Pre-Training of Audio Transformers for Sound Event Detection**](https://arxiv.org/abs/2409.09546). 

---

# Setting up the Environment

1. If needed, create a new environment with python 3.9 and activate it:

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

4. Install package for mp3 decoding:

``` bash
CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip
```


### ! Repository in process !

This repository is in process and will be updated with the pre-trained models and the code for training the efficient models in the next few weeks.

**Update:** Repository is constantly being updated - this is not the final version.

**TODO:**
- [x] Add code for all the sequence models.
- [x] Add code for training the efficient CNNs on AudioSet Strong.
- [x] Merge the two Wrapper classes for the Audio Transformers and the Efficient CNNs.
- [ ] Add file to train the efficient models using the "Online Teacher KD" setup. 
- [x] Add pre-trained models.
- [x] Check inference.py for the efficient models.
- [ ] Add plots from the paper to the README file. 
- [ ] Adapt ex_dcase2016task2.py to work with the efficient models.
- [x] Update requirements.txt





# References

[1] A. Howard, R. Pang, H. Adam, Q. V. Le, M. Sandler, B. Chen, W. Wang, L. Chen, M. Tan, G. Chu, V. Vasudevan, and Y. Zhu, “Searching for mobilenetv3,” in Proceedings of the International Conference on Computer Vision (ICCV), 2019.