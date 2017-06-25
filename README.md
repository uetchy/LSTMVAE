# LSTMVAE
LSTMVAE and LSTMCVAE implemented with chainer.

This code is for Python 3.
needs some Library as follows.
  chainer 2.0
  cupy 1.0(if you use GPU)
  numpy>=1.12

Codes are based on
[Generating sentences from continuous space.](https://arxiv.org/abs/1511.06349)
written by Samuel R. Bowman in 2015.

## Pre-trained Model
Models can be downloaded from
VAE
```
. ./src/each_case/prof/model/download_model.sh
```
CVAE
```
. ./src/each_case/serif/model/download_model.sh
```
Test can be executed by
VAE
```
python ./src/each_case/sampleVAEProf.py
```
CVAE
```
python ./src/each_case/sampleCVAESerif.py
```

Hyperparameters are also set there.

## Train and Test
As to VAE, training data are in
./src/each_case/prof/prof16000fixed_longcut.txt

As to CVAE, training data cannnot be uploaded due to the authority problems.
Dummy data are in 
./src/each_case/serif/all_serif16000_fixed_dummy.txt
./src/each_case/serif/all_chara_dummy.txt

Training can be executed by
VAE
```
python ./src/each_case/sampleVAEProf.py --train
```
CVAE
```
python ./src/each_case/sampleCVAESerif.py --train
```
Hyperparameters are also set there.

Then, test can be executed as follows. 
VAE
```
python ./src/each_case/sampleVAEProf.py
```
CVAE
```
python ./src/each_case/sampleCVAESerif.py
```
