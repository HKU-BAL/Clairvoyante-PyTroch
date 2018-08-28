#!/bin/bash

# Trains the Clairvoyante Pytorch v3 model as a daemon process.
# --ochk_prefix ../pytorchModels/trainAll4_1000PGPU/v4

nohup python train.py --bin_fn /nas7/yswong/trainingData/tensor_all.bin & 2> nohup.out
echo $! > save_pid.txt
