#!/bin/bash

# Trains the Clairvoyante Pytorch v3 model as a daemon process.

git pull
nohup python train.py & 2> nohup.out
echo $! > save_pid.txt
