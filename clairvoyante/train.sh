#!/bin/bash

# Trains the Clairvoyante Pytorch v3 model as a daemon process.

git pull
nohup python train.py --bin_fn /nas7/yswong/trainingData/tensor_all.bin --ochk_prefix ../pytorchModels/trainAll2 & 2> nohup.out
echo $! > save_pid.txt
