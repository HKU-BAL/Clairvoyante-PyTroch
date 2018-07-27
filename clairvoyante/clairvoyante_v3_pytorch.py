# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classifies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import param

class Net(nn.Module):

    # Initialises Clairvoyante with 3 convolutional layers, 2 hidden fully connected layers and an output layer.
    # It specifies the parameters for these layers and it initialises the NN's weights using He initializtion.
    def __init__(self, inputShape = (2*param.flankingBaseNum+1, 4, param.matrixNum),
                       outputShape1 = (4, ), outputShape2 = (2, ), outputShape3 = (4, ), outputShape4 = (6, ),
                       kernelSize1 = (1, 4), kernelSize2 = (2, 4), kernelSize3 = (3, 4),
                       pollSize1 = (5, 1), pollSize2 = (4, 1), pollSize3 = (3, 1),
                       numFeature1 = 16, numFeature2 = 32, numFeature3 = 48,
                       hiddenLayerUnits4 = 336, hiddenLayerUnits5 = 168,
                       initialLearningRate = param.initialLearningRate, learningRateDecay = param.learningRateDecay,
                       dropoutRateFC4 = param.dropoutRateFC4, dropoutRateFC5 = param.dropoutRateFC5,
                       l2RegularizationLambda = param.l2RegularizationLambda, l2RegularizationLambdaDecay = param.l2RegularizationLambdaDecay):
        super(Net, self).__init__()

        self.inputShape = inputShape
        self.outputShape1 = outputShape1; self.outputShape2 = outputShape2; self.outputShape3 = outputShape3; self.outputShape4 = outputShape4
        self.kernelSize1 = kernelSize1; self.kernelSize2 = kernelSize2; self.kernelSize3 = kernelSize3
        self.pollSize1 = pollSize1; self.pollSize2 = pollSize2; self.pollSize3 = pollSize3
        self.numFeature1 = numFeature1; self.numFeature2 = numFeature2; self.numFeature3 = numFeature3
        self.hiddenLayerUnits4 = hiddenLayerUnits4; self.hiddenLayerUnits5 = hiddenLayerUnits5
        self.learningRateVal = initialLearningRate; self.learningRateDecay = learningRateDecay
        self.dropoutRateFC4Val = dropoutRateFC4; self.dropoutRateFC5Val = dropoutRateFC5
        self.l2RegularizationLambdaVal = l2RegularizationLambda; self.l2RegularizationLambdaDecay = l2RegularizationLambdaDecay
        self.trainLossRTVal = None; self.trainSummaryRTVal = None; self.getLossLossRTVal = None
        self.predictBaseRTVal = None; self.predictZygosityRTVal = None; self.predictVarTypeRTVal = None; self.predictIndelLengthRTVal = None

        # 3 Convolutional Layers
        # channel = int(self.inputShape[1])
        self.conv1 = nn.Conv2d(param.matrixNum, self.numFeature1, self.kernelSize1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')

        self.conv2 = nn.Conv2d(self.numFeature1, self.numFeature2, self.kernelSize2)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

        self.conv3 = nn.Conv2d(self.numFeature2, self.numFeature3, self.kernelSize3)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')

        # Calculate the size of the flattened size after the conv3
        self.flat_size = ( self.inputShape[0] - (self.pollSize1[0] - 1) - (self.pollSize2[0] - 1) - (self.pollSize3[0] - 1))
        self.flat_size *= ( self.inputShape[1] - (self.pollSize1[1] - 1) - (self.pollSize2[1] - 1) - (self.pollSize3[1] - 1))
        self.flat_size *= self.numFeature3

        # 2 FC Hidden Layers
        self.fc4 = nn.Linear(self.flat_size, self.hiddenLayerUnits4)
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')

        self.fc5 = nn.Linear(self.hiddenLayerUnits4, self.hiddenLayerUnits5)
        nn.init.kaiming_normal_(self.fc5.weight, mode='fan_in', nonlinearity='relu')

        # 4 Output Layers
        self.YBaseChangeSigmoidLayer = nn.Linear(self.hiddenLayerUnits4,self.outputShape1[0])
        nn.init.kaiming_normal_(self.YBaseChangeSigmoidLayer.weight, mode='fan_in', nonlinearity='relu')

        self.YZygosityFCLayer = nn.Linear(self.hiddenLayerUnits5,self.outputShape2[0])
        nn.init.kaiming_normal_(self.YZygosityFCLayer.weight, mode='fan_in', nonlinearity='relu')

        self.YVarTypeFCLayer = nn.Linear(self.hiddenLayerUnits5, self.outputShape3[0])
        nn.init.kaiming_normal_(self.YVarTypeFCLayer.weight, mode='fan_in', nonlinearity='relu')

        self.YIndelLengthFCLayer = nn.Linear(self.hiddenLayerUnits5, self.outputShape4[0])
        nn.init.kaiming_normal_(self.YIndelLengthFCLayer.weight, mode='fan_in', nonlinearity='relu')

    # Implements the same padding feature in Tensorflow.
    # kernelSize is a tuple as kernel is not a square.
    def padding(self, kernelSize):
        ka1 = kernelSize[0] // 2
        kb1 = ka1 - 1 if kernelSize[0] % 2 == 0 else ka1
        ka2 = kernelSize[1] // 2
        kb2 = ka2 - 1 if kernelSize[1] % 2 == 0 else ka2
        print((kb2,ka2,kb1,ka1))
        return nn.ZeroPad2d((kb2,ka2,kb1,ka1))

    # Forward propagation
    def forward(self, XPH):
        # Different non-linear activation functions.
        selu = nn.SELU()
        sigmoid = nn.Sigmoid()
        # Dim specifies softmax over the row of the tensor.
        softmax = nn.Softmax(dim=1)

        # Convolution layers with max max_pooling and SELU.
        pad1 = self.padding(self.kernelSize1)
        print(selu(self.conv1(pad1(XPH))).shape)
        pool1 = F.max_pool2d(selu(self.conv1(pad1(XPH))), self.pollSize1, stride=1)
        print(pool1.shape)

        pad2 = self.padding(self.kernelSize2)
        print(selu(self.conv2(pad2(pool1))).shape)
        pool2 = F.max_pool2d(selu(self.conv2(pad2(pool1))), self.pollSize2, stride=1)
        print(pool2.shape)

        pad3 = self.padding(self.kernelSize3)
        print(selu(self.conv3(pad3(pool2))).shape)
        pool3 = F.max_pool2d(selu(self.conv3(pad3(pool2))), self.pollSize3, stride=1)
        print(pool3.shape)

        # Flattens output from pool3.
        conv3_flat = pool3.view(-1, self.flat_size)
        print(conv3_flat.shape)

        # 2 Hidden Layers that uses SELU and alpha dropout.
        # Alphadropout uses bernoulli distribution rather than uniform distribution.
        selu4 = selu(self.fc4(conv3_flat))
        ad4 = nn.AlphaDropout(p=self.dropoutRateFC4Val)
        dropout4 = ad4(selu4)
        print(dropout4.shape)

        selu5 = selu(self.fc5(dropout4))
        ad5 = nn.AlphaDropout(p=self.dropoutRateFC5Val)
        dropout5 = ad5(selu5)
        print(dropout5.shape)

        # Epsilon for softmax.
        epsilon = 1e-10

        # 1 output layer that uses sigmoid for base change.
        YBaseChangeSigmoid = sigmoid(self.YBaseChangeSigmoidLayer(dropout4))
        self.YBaseChangeSigmoid = YBaseChangeSigmoid
        print(YBaseChangeSigmoid.shape)

        # 3 output fully connected layers for zygosity, varType and indelLength.
        # Uses SELU and softmax to output result.
        YZygosityFC = selu(self.YZygosityFCLayer(dropout5))
        YZygosityLogits = torch.add(YZygosityFC, epsilon)
        self.YZygosityLogits = YZygosityLogits
        YZygositySoftmax = softmax(YZygosityLogits)
        self.YZygositySoftmax = YZygositySoftmax
        print(YZygositySoftmax.shape)

        YVarTypeFC = selu(self.YVarTypeFCLayer(dropout5))
        YVarTypeLogits = torch.add(YVarTypeFC, epsilon)
        self.YVarTypeLogits = YVarTypeLogits
        YVarTypeSoftmax = softmax(YVarTypeLogits)
        self.YVarTypeSoftmax = YVarTypeSoftmax
        print(YVarTypeSoftmax.shape)

        YIndelLengthFC = selu(self.YIndelLengthFCLayer(dropout5))
        YIndelLengthLogits = torch.add(YIndelLengthFC, epsilon)
        self.YIndelLengthLogits = YIndelLengthLogits
        YIndelLengthSoftmax = softmax(YIndelLengthLogits)
        self.YIndelLengthSoftmax = YIndelLengthSoftmax
        print(YIndelLengthSoftmax.shape)

        return YBaseChangeSigmoid,YZygositySoftmax,YVarTypeSoftmax,YIndelLengthSoftmax

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


net = Net()
print(net)

########################################################################
# You just have to define the ``forward`` function, and the ``backward``
# function (where gradients are computed) is automatically defined for you
# using ``autograd``.
# You can use any of the Tensor operations in the ``forward`` function.
#
# The learnable parameters of a model are returned by ``net.parameters()``

params = list(net.parameters())
print(len(params))
# print(params[0].size())  # conv1's .weight

########################################################################
# Let try a random 32x32 input
# Note: Expected input size to this net(LeNet) is 32x32. To use this net on
# MNIST dataset, please resize the images from the dataset to 32x32.

input = torch.randn(param.matrixNum, 2*param.flankingBaseNum+1, 4)
print(input.shape)
out = net(input.unsqueeze_(0))
# print(out)

# ########################################################################
# # Zero the gradient buffers of all parameters and backprops with random
# # gradients:
# net.zero_grad()
# out.backward(torch.randn(1, 10))
#
########################################################################
# .. note::
#
#     ``torch.nn`` only supports mini-batches. The entire ``torch.nn``
#     package only supports inputs that are a mini-batch of samples, and not
#     a single sample.
#
#     For example, ``nn.Conv2d`` will take in a 4D Tensor of
#     ``nSamples x nChannels x Height x Width``.
#
#     If you have a single sample, just use ``input.unsqueeze(0)`` to add
#     a fake batch dimension.
#
# Before proceeding further, let's recap all the classes you’ve seen so far.
#
# **Recap:**
#   -  ``torch.Tensor`` - A *multi-dimensional array* with support for autograd
#      operations like ``backward()``. Also *holds the gradient* w.r.t. the
#      tensor.
#   -  ``nn.Module`` - Neural network module. *Convenient way of
#      encapsulating parameters*, with helpers for moving them to GPU,
#      exporting, loading, etc.
#   -  ``nn.Parameter`` - A kind of Tensor, that is *automatically
#      registered as a parameter when assigned as an attribute to a*
#      ``Module``.
#   -  ``autograd.Function`` - Implements *forward and backward definitions
#      of an autograd operation*. Every ``Tensor`` operation, creates at
#      least a single ``Function`` node, that connects to functions that
#      created a ``Tensor`` and *encodes its history*.
#
# **At this point, we covered:**
#   -  Defining a neural network
#   -  Processing inputs and calling backward
#
# **Still Left:**
#   -  Computing the loss
#   -  Updating the weights of the network
#
# Loss Function
# -------------
# A loss function takes the (output, target) pair of inputs, and computes a
# value that estimates how far away the output is from the target.
#
# There are several different
# `loss functions <http://pytorch.org/docs/nn.html#loss-functions>`_ under the
# nn package .
# A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
# between the input and the target.
#
# For example:
#
# output = net(input)
YPH = torch.randn(1, 16)

print("LOSS")
# loss1 = (net.YBaseChangeSigmoid - YPH.narrow(1, 0, net.outputShape1[0])).pow(2).sum()
# print(loss1)

# Calculates MSE without computing average.
mse = nn.MSELoss(size_average=False)
loss1 = mse(net.YBaseChangeSigmoid, YPH.narrow(1, 0, net.outputShape1[0]))
print(loss1)

log_softmax = nn.LogSoftmax(dim=1)
print(net.YZygosityLogits)

YZygosityCrossEntropy = log_softmax(net.YZygosityLogits) * -YPH.narrow(1, net.outputShape1[0], net.outputShape2[0])
loss2 = YZygosityCrossEntropy.sum()
print(loss2)

YVarTypeCrossEntropy = log_softmax(net.YVarTypeLogits) * -YPH.narrow(1, net.outputShape1[0]+net.outputShape2[0], net.outputShape3[0])
loss3 = YVarTypeCrossEntropy.sum()
print(loss3)

YIndelLengthCrossEntropy = log_softmax(net.YIndelLengthLogits) * -YPH.narrow(1, net.outputShape1[0]+net.outputShape2[0]+net.outputShape3[0], net.outputShape4[0])
loss4 = YIndelLengthCrossEntropy.sum()
print(loss4)

l2_reg = None
for name, W in net.named_parameters():
    if name not in ['bias']:
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
print(l2_reg)

lossL2 = l2_reg * net.l2RegularizationLambdaVal
print(lossL2)

loss = loss1 + loss2 + loss3 + loss4 + lossL2
# loss = loss1 + loss2 + loss3 + loss4
net.loss = loss
print(loss)
#
# ########################################################################
# # Now, if you follow ``loss`` in the backward direction, using its
# # ``.grad_fn`` attribute, you will see a graph of computations that looks
# # like this:
# #
# # ::
# #
# #     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
# #           -> view -> linear -> relu -> linear -> relu -> linear
# #           -> MSELoss
# #           -> loss
# #
# # So, when we call ``loss.backward()``, the whole graph is differentiated
# # w.r.t. the loss, and all Tensors in the graph that has ``requires_grad=True``
# # will have their ``.grad`` Tensor accumulated with the gradient.
# #
# # For illustration, let us follow a few steps backward:
#
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
#
# ########################################################################
# # Backprop
# # --------
# # To backpropagate the error all we have to do is to ``loss.backward()``.
# # You need to clear the existing gradients though, else gradients will be
# # accumulated to existing gradients.
# #
# #
# # Now we shall call ``loss.backward()``, and have a look at conv1's bias
# # gradients before and after the backward.
#
#
# net.zero_grad()     # zeroes the gradient buffers of all parameters
#
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
#
# loss.backward()
#
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
#
########################################################################
# Now, we have seen how to use loss functions.
#
# **Read Later:**
#
#   The neural network package contains various modules and loss functions
#   that form the building blocks of deep neural networks. A full list with
#   documentation is `here <http://pytorch.org/docs/nn>`_.
#
# **The only thing left to learn is:**
#
#   - Updating the weights of the network
#
# Update the weights
# ------------------
# The simplest update rule used in practice is the Stochastic Gradient
# Descent (SGD):
#
#      ``weight = weight - learning_rate * gradient``
#
# We can implement this using simple python code:
#
# .. code:: python
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# However, as you use neural networks, you want to use various different
# update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we built a small package: ``torch.optim`` that
# implements all these methods. Using it is very simple:

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=net.learningRateVal)

# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# for x in range(10):
#
#     print('conv1.bias.grad before backward')
#     print(net.conv1.bias.grad)
#
#     loss.backward()
#
#     print('conv1.bias.grad after backward')
#     print(net.conv1.bias.grad)
#     print(net.conv1.bias.shape)
#     print(loss)
#
# optimizer.step()    # Does the update


###############################################################
# .. Note::
#
#       Observe how gradient buffers had to be manually set to zero using
#       ``optimizer.zero_grad()``. This is because gradients are accumulated
#       as explained in `Backprop`_ section.
