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

# Saves the weights and biases in fullv3-illumina-novoalign-hg001+hg002-hg38 into files.

if __name__ == "__main__":
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters('../trainedModels/fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e-3.epoch500')

    for variable in tf.trainable_variables():
        print(variable)
        print("\n")
