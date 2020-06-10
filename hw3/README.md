# README

## Requirements

* python 3.6
* pytorch 1.3.0
* scikit-learn 0.20.1
* numpy 1.15.4
* matplotlib
* pickle



## Files

* **119034910071_NN_assignment3.pdf**: final report
* **code**: code file
  * **dataset**
    * **SEED.py**: SEED dataset loader
  * **model**
    * **DANN.py**: DANN model
    * **MDAN.py**: MDAN model
    * **newlayer.py**: implementation of gradient reversal layer
  * **run**
    * **test.py**: define test function
  * **train\_baseline.py/train\_dann.py/train\_mdan.py**: train file
  * **vis.py**: data distribution visualization file



## Run

1. Move the `data.pkl` file to `code` directory;
2. create a directory `models` in `code` directory to save models;
3. run `CUDA_VISIBLE_DEVICES=$GPU_ID python train_xxx.py` to train and test the ccoresponding model.