import sys
import time
import argparse
import param
import logging
import numpy as np
import utils_v2 as utils
import torch
# import clairvoyante_v2 as cv
import clairvoyante_v3 as cv
import clairvoyante_v3_pytorch as cpt
import tensorflow as tf
import os
import h5py
from ast import literal_eval as make_tuple

# Loads the weights and biases into the pytorch model.

if __name__ == "__main__":
    mp = cpt.Net()

    # Gets each param in folder and load it into pytorch model.
    for filename in os.listdir('../illumina_2_parameters/h5'):
        par = h5py.File('../illumina_2_parameters/h5/' + filename+'.h5', 'r')
        par = torch.from_numpy(par['weights'][()])

        # Gets rid of .h5 extension.
        name_list = filename[:-3].split('_')

        if name_list[1] == "kernel":
            name_list[1] = "weight"

        if name_list[0][0] == 'Y':
            name_list[0] += "Layer"

        par_name = name_list[0] + "." + name_list[1]
        dimension = make_tuple(name_list[2])

        # print(par_name)
        # print(dimension)
        # print(par.reshape(dimension))
        # for param in mp.parameters():
        #     print(param)
        #
        # for W in mp.parameters():
        #     # Conv Weights
        if "conv" in par_name and "weight" in par_name:
            # print(par_name)
            # print(name)
            # print(W)
            # print(dimension)
            # print(par.reshape(dimension))
            W = par.permute(3,2,0,1)
            # print(W)
            # print(W.shape)
            # break
        # Biases
        elif "bias" in par_name:
            # print(par_name)
            # print(name)
            # print(W)
            # print(dimension)
            # print(par.reshape(dimension))
            W = par
            # print(W)
            # print(W.shape)
            # break
        # FC weights
        elif "weight" in par_name:
            # print(par_name)
            # print(name)
            # print(W)
            # print(dimension)
            # print(par.reshape(dimension))
            W = par.t()
            # print(W)
            # print(W.shape)
            # break
        # print("\n")
        print(par_name)
        print(W.shape)
        print(W)
        print("\n")
        mp.state_dict()[par_name].data.copy_(W)

    print("\n")
    print("Actual params: ")
    for name, W in mp.named_parameters():
        print(name)
        print(W.shape)
        print(W)
        print("\n")

    torch.save(mp.state_dict(), "../pytorchModels/illumina_2_h5/illumina_2_parameters.txt")
