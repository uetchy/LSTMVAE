# LSTMVAE
LSTMVAE implemented with chainer.

This code is for Python 3.

Codes are based on
[Generating sentences from continuous space.](https://arxiv.org/abs/1511.06349)
written by Samuel R. Bowman in 2015.

## Pre-trained Model
Models can be downloaded from
```
. ./src/each_case/prof/model/download_model.sh
```
If the file name is incorrect, please rename it to "biconcatlstm_vae_kl_prof_29_l200.npz".

Test can be executed by
```
python ./src/each_case/sampleVAEProf.py
```
Hyperparameters are also set there.

## Train and Test
Training data are in
.src/each_case/prof/prof16000fixed_longcut.txt

Training can be executed by
```
python ./src/each_case/sampleVAEProf.py --train
```
Then, test can be executed as above. Hyperparameters are also set there.
