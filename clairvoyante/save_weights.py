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

# Saves the weights and biases in fullv3-illumina-novoalign-hg001+hg002-hg38 into files.

if __name__ == "__main__":
    # utils.SetupEnv()
    # m = cv.Clairvoyante()
    # m.init()
    #
    # m.restoreParameters('../trainedModels/fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e-3.epoch500')

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
        print(par_name)
        print(dimension)
