import sys
import os
import time
import argparse
import param
import logging
import pickle
import numpy as np
from threading import Thread
import torch
import torch.nn as nn
import torch.nn.functional as F
import param
import sys
import torch.optim as optim

def MSE(sigmoid, YPH):
    mse = nn.MSELoss(reduction='sum')
    loss = mse(sigmoid, YPH)

    return loss.data.numpy()

def CEL(logits, YPH):
    log_softmax = nn.LogSoftmax(dim=0)
    CrossEntropy = log_softmax(logits) * -YPH
    print(CrossEntropy)
    loss = CrossEntropy.sum()

    return loss.data.numpy()

def MSETest():
    # Test 1
    x = torch.tensor([[1., 1.],[1., 1.]], requires_grad=True)
    y = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
    loss = MSE(x,y)
    print("X: " + str([[1, 1],[1,1]]))
    print("Y: " + str([[0,0],[0,0]]))
    print(str(loss))
    assert(loss == 4)
    print("Success \n")

    # Test 2
    x = torch.tensor([[1., 1.],[1., 1.]], requires_grad=True)
    y = torch.tensor([[1.,1.],[1.,1.]], requires_grad=True)
    loss = MSE(x,y)
    print("X: " + str([[1, 1],[1,1]]))
    print("Y: " + str([[1,1],[1,1]]))
    print(str(loss))
    assert(loss == 0)
    print("Success \n")

    # Test 3
    x = torch.tensor([[3., -1.],[-1., 3.]], requires_grad=True)
    y = torch.tensor([[-2.,-5.],[-6.,7.]], requires_grad=True)
    loss = MSE(x,y)
    print("X: " + str([[3., -1.],[-1.,3.]]))
    print("Y: " + str([[-2.,-5.],[-6.,7.]]))
    print(str(loss))
    assert(loss == 82)
    print("Success \n")

def CELTest():
    # Test 1
    x = torch.tensor([[1.,1.],[1.,1.]], requires_grad=True)
    y = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
    print("X: " + str([[1, 1],[1,1]]))
    print("Y: " + str([[0,0],[0,0]]))
    loss = CEL(x,y)
    print(str(loss))
    assert(loss == 0)
    print("Success \n")

    # Test 2
    x = torch.tensor([[10.,10.],[10.,10.]], requires_grad=True)
    y = torch.tensor([[10.,-7.],[-3.,-4.]], requires_grad=True)
    print("X: " + str([[10, 10],[10,10]]))
    print("Y: " + str([[10.,-7.],[-3.,-4.]]))
    loss = CEL(x,y)
    loss = round(loss,7)
    print(str(loss))
    assert(loss == -2.7725906)
    print("Success \n")

if __name__ == "__main__":
    MSETest()
    CELTest()
