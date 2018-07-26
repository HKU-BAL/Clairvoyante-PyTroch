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

        # 3 convolutional layers and 3 fully connected layers.
        # channel = int(self.inputShape[1])
        # kernel
        self.conv1 = nn.Conv2d(param.matrixNum, self.numFeature1, self.kernelSize1)
        self.conv2 = nn.Conv2d(self.numFeature1, self.numFeature2, self.kernelSize2)
        self.conv3 = nn.Conv2d(self.numFeature2, self.numFeature3, self.kernelSize3)

        # Calculate the size of the flattened size after the conv3
        self.flat_size = ( self.inputShape[0] - (self.pollSize1[0] - 1) - (self.pollSize2[0] - 1) - (self.pollSize3[0] - 1))
        self.flat_size *= ( self.inputShape[1] - (self.pollSize1[1] - 1) - (self.pollSize2[1] - 1) - (self.pollSize3[1] - 1))
        self.flat_size *= self.numFeature3

        # an affine operation: y = Wx + b
        self.fc4 = nn.Linear(self.flat_size, self.hiddenLayerUnits4)
        self.fc5 = nn.Linear(self.hiddenLayerUnits4, self.hiddenLayerUnits5)
        self.fc3 = nn.Linear(self.hiddenLayerUnits5, 12)

    # Implements the same padding feature in Tensorflow.
    # KernelSize is a tuple as kernel is not a square.
    def padding(self, kernelSize):
        ka1 = kernelSize[0] // 2
        kb1 = ka1 - 1 if kernelSize[0] % 2 == 0 else ka1
        ka2 = kernelSize[1] // 2
        kb2 = ka2 - 1 if kernelSize[1] % 2 == 0 else ka2
        print((kb2,ka2,kb1,ka1))
        return((kb2,ka2,kb1,ka1))

    def forward(self, x):
        # Max pooling over a self.pollSize1 window
        selu = nn.SELU()

        # pad1 = nn.ZeroPad2d((1,2,0,0))
        pad1 = nn.ZeroPad2d(self.padding(self.kernelSize1))
        print(selu(self.conv1(pad1(x))).shape)
        pool1 = F.max_pool2d(selu(self.conv1(pad1(x))), self.pollSize1, stride=1)
        print(pool1.shape)

        # If the size is a square you can only specify a single number
        pad2 = nn.ZeroPad2d(self.padding(self.kernelSize2))
        print(selu(self.conv2(pad2(pool1))).shape)
        pool2 = F.max_pool2d(selu(self.conv2(pad2(pool1))), self.pollSize2, stride=1)
        print(pool2.shape)

        pad3 = nn.ZeroPad2d(self.padding(self.kernelSize3))
        print(selu(self.conv3(pad3(pool2))).shape)
        pool3 = F.max_pool2d(selu(self.conv3(pad3(pool2))), self.pollSize3, stride=1)
        print(pool3.shape)

        conv3_flat = pool3.view(-1, self.flat_size)
        print(conv3_flat.shape)

        dropout4 = selu(self.fc4(conv3_flat))
        print(dropout4.shape)

        dropout5 = selu(self.fc5(dropout4))
        print(dropout5.shape)
        
        # dropout6 = self.fc3(dropout5)
        #
        # return dropout6

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


net = Net()
print(net)

# ########################################################################
# # You just have to define the ``forward`` function, and the ``backward``
# # function (where gradients are computed) is automatically defined for you
# # using ``autograd``.
# # You can use any of the Tensor operations in the ``forward`` function.
# #
# # The learnable parameters of a model are returned by ``net.parameters()``
#
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight
#
# ########################################################################
# # Let try a random 32x32 input
# # Note: Expected input size to this net(LeNet) is 32x32. To use this net on
# # MNIST dataset, please resize the images from the dataset to 32x32.
#
input = torch.randn(param.matrixNum, 2*param.flankingBaseNum+1, 4)
print(input.shape)
out = net(input.unsqueeze_(0))
print(out)
#
# ########################################################################
# # Zero the gradient buffers of all parameters and backprops with random
# # gradients:
# net.zero_grad()
# out.backward(torch.randn(1, 10))
#
# ########################################################################
# # .. note::
# #
# #     ``torch.nn`` only supports mini-batches. The entire ``torch.nn``
# #     package only supports inputs that are a mini-batch of samples, and not
# #     a single sample.
# #
# #     For example, ``nn.Conv2d`` will take in a 4D Tensor of
# #     ``nSamples x nChannels x Height x Width``.
# #
# #     If you have a single sample, just use ``input.unsqueeze(0)`` to add
# #     a fake batch dimension.
# #
# # Before proceeding further, let's recap all the classes you’ve seen so far.
# #
# # **Recap:**
# #   -  ``torch.Tensor`` - A *multi-dimensional array* with support for autograd
# #      operations like ``backward()``. Also *holds the gradient* w.r.t. the
# #      tensor.
# #   -  ``nn.Module`` - Neural network module. *Convenient way of
# #      encapsulating parameters*, with helpers for moving them to GPU,
# #      exporting, loading, etc.
# #   -  ``nn.Parameter`` - A kind of Tensor, that is *automatically
# #      registered as a parameter when assigned as an attribute to a*
# #      ``Module``.
# #   -  ``autograd.Function`` - Implements *forward and backward definitions
# #      of an autograd operation*. Every ``Tensor`` operation, creates at
# #      least a single ``Function`` node, that connects to functions that
# #      created a ``Tensor`` and *encodes its history*.
# #
# # **At this point, we covered:**
# #   -  Defining a neural network
# #   -  Processing inputs and calling backward
# #
# # **Still Left:**
# #   -  Computing the loss
# #   -  Updating the weights of the network
# #
# # Loss Function
# # -------------
# # A loss function takes the (output, target) pair of inputs, and computes a
# # value that estimates how far away the output is from the target.
# #
# # There are several different
# # `loss functions <http://pytorch.org/docs/nn.html#loss-functions>`_ under the
# # nn package .
# # A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
# # between the input and the target.
# #
# # For example:
#
# output = net(input)
# target = torch.arange(1, 11)  # a dummy target, for example
# target = target.view(1, -1)  # make it the same shape as output
# criterion = nn.MSELoss()
#
# loss = criterion(output, target)
# print(loss)
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
# ########################################################################
# # Now, we have seen how to use loss functions.
# #
# # **Read Later:**
# #
# #   The neural network package contains various modules and loss functions
# #   that form the building blocks of deep neural networks. A full list with
# #   documentation is `here <http://pytorch.org/docs/nn>`_.
# #
# # **The only thing left to learn is:**
# #
# #   - Updating the weights of the network
# #
# # Update the weights
# # ------------------
# # The simplest update rule used in practice is the Stochastic Gradient
# # Descent (SGD):
# #
# #      ``weight = weight - learning_rate * gradient``
# #
# # We can implement this using simple python code:
# #
# # .. code:: python
# #
# #     learning_rate = 0.01
# #     for f in net.parameters():
# #         f.data.sub_(f.grad.data * learning_rate)
# #
# # However, as you use neural networks, you want to use various different
# # update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
# # To enable this, we built a small package: ``torch.optim`` that
# # implements all these methods. Using it is very simple:
#
# import torch.optim as optim
#
# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)
#
# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # Does the update
#
#
# ###############################################################
# # .. Note::
# #
# #       Observe how gradient buffers had to be manually set to zero using
# #       ``optimizer.zero_grad()``. This is because gradients are accumulated
# #       as explained in `Backprop`_ section.
