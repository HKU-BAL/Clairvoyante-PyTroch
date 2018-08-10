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
from ast import literal_eval as make_tuple

# Loads the weights and biases into the pytorch model.

if __name__ == "__main__":
    mp = cpt.Net()

    # Gets each param in folder and load it into pytorch model.
    for filename in os.listdir('../illumina_2_parameters/'):
        par = np.loadtxt('../illumina_2_parameters/' + filename)

        # Gets rid of .txt extension.
        name_list = filename[:-4].split('_')

        if name_list[1] == "kernel":
            name_list[1] = "weight"

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
            W = torch.from_numpy(par.reshape(dimension)).permute(0,3,1,2)
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
            W = torch.from_numpy(par.reshape(dimension))
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
            W = torch.from_numpy(par.reshape(dimension)).permute(0,1)
            # print(W)
            # print(W.shape)
            # break
        # print("\n")
        print(name)
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

    #torch.save(mp.state_dict(), "../pytorchModels/illumina_2/illumina_2_parameters.txt")
