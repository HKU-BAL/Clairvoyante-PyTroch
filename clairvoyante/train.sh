#!/bin/bash

# Trains the Clairvoyante Pytorch v3 model as a daemon process.

git pull
nohup python train.py --bin_fn ../training/tensor.bin & 2> nohup.out
echo $! > save_pid.txt
