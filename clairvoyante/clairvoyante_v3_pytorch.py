# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import param
import torch.optim as optim

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

        self.optimizer = optim.Adam(self.parameters(), lr=self.learningRateVal)

    # Implements the same padding feature in Tensorflow.
    # kernelSize is a tuple as kernel is not a square.
    def padding(self, kernelSize):
        ka1 = kernelSize[0] // 2
        kb1 = ka1 - 1 if kernelSize[0] % 2 == 0 else ka1
        ka2 = kernelSize[1] // 2
        kb2 = ka2 - 1 if kernelSize[1] % 2 == 0 else ka2
        # print((kb2,ka2,kb1,ka1))
        return nn.ZeroPad2d((kb2,ka2,kb1,ka1))

    # Forward propagation
    def forward(self, XPH):
        # print(XPH)

        # Different non-linear activation functions.
        selu = nn.SELU()
        sigmoid = nn.Sigmoid()
        # Dim specifies softmax over the row of the tensor.
        softmax = nn.Softmax(dim=1)

        # Convolution layers with max max_pooling and SELU.
        pad1 = self.padding(self.kernelSize1)
        # print(selu(self.conv1(pad1(XPH))).shape)
        pool1 = F.max_pool2d(selu(self.conv1(pad1(XPH))), self.pollSize1, stride=1)
        # print(pool1.shape)

        pad2 = self.padding(self.kernelSize2)
        # print(selu(self.conv2(pad2(pool1))).shape)
        pool2 = F.max_pool2d(selu(self.conv2(pad2(pool1))), self.pollSize2, stride=1)
        # print(pool2.shape)

        pad3 = self.padding(self.kernelSize3)
        # print(selu(self.conv3(pad3(pool2))).shape)
        pool3 = F.max_pool2d(selu(self.conv3(pad3(pool2))), self.pollSize3, stride=1)
        # print(pool3.shape)

        # Flattens output from pool3.
        conv3_flat = pool3.view(-1, self.flat_size)
        # print(conv3_flat.shape)

        # 2 Hidden Layers that uses SELU and alpha dropout.
        # Alphadropout uses bernoulli distribution rather than uniform distribution.
        selu4 = selu(self.fc4(conv3_flat))
        ad4 = nn.AlphaDropout(p=self.dropoutRateFC4Val)
        dropout4 = ad4(selu4)
        # print(dropout4.shape)

        selu5 = selu(self.fc5(dropout4))
        ad5 = nn.AlphaDropout(p=self.dropoutRateFC5Val)
        dropout5 = ad5(selu5)
        # print(dropout5.shape)

        # Epsilon for softmax.
        epsilon = 1e-10

        # 1 output layer that uses sigmoid for base change.
        YBaseChangeSigmoid = sigmoid(self.YBaseChangeSigmoidLayer(dropout4))
        self.YBaseChangeSigmoid = YBaseChangeSigmoid
        # print(YBaseChangeSigmoid.shape)

        # 3 output fully connected layers for zygosity, varType and indelLength.
        # Uses SELU and softmax to output result.
        YZygosityFC = selu(self.YZygosityFCLayer(dropout5))
        YZygosityLogits = torch.add(YZygosityFC, epsilon)
        self.YZygosityLogits = YZygosityLogits
        # print(YZygosityLogits)
        YZygositySoftmax = softmax(YZygosityLogits)
        self.YZygositySoftmax = YZygositySoftmax
        # print(YZygositySoftmax.shape)

        YVarTypeFC = selu(self.YVarTypeFCLayer(dropout5))
        YVarTypeLogits = torch.add(YVarTypeFC, epsilon)
        self.YVarTypeLogits = YVarTypeLogits
        # print(YVarTypeLogits)
        YVarTypeSoftmax = softmax(YVarTypeLogits)
        self.YVarTypeSoftmax = YVarTypeSoftmax
        # print(YVarTypeSoftmax.shape)

        YIndelLengthFC = selu(self.YIndelLengthFCLayer(dropout5))
        YIndelLengthLogits = torch.add(YIndelLengthFC, epsilon)
        self.YIndelLengthLogits = YIndelLengthLogits
        # print(YIndelLengthLogits)
        YIndelLengthSoftmax = softmax(YIndelLengthLogits)
        self.YIndelLengthSoftmax = YIndelLengthSoftmax
        # print(YIndelLengthSoftmax.shape)

        return YBaseChangeSigmoid.data.numpy(),YZygositySoftmax.data.numpy(),YVarTypeSoftmax.data.numpy(),YIndelLengthSoftmax.data.numpy()

    def costFunction(self, YPH):
        # Calculates MSE without computing average.
        mse = nn.MSELoss(size_average=False)
        loss1 = mse(net.YBaseChangeSigmoid, YPH.narrow(1, 0, net.outputShape1[0]))
        print("Loss1: "+str(loss1)+"\n")

        log_softmax = nn.LogSoftmax(dim=1)

        print(net.YZygosityLogits)
        YZygosityCrossEntropy = log_softmax(net.YZygosityLogits) * -YPH.narrow(1, net.outputShape1[0], net.outputShape2[0])
        print(YZygosityCrossEntropy)
        loss2 = YZygosityCrossEntropy.sum()
        print("Loss2: "+str(loss2)+"\n")

        print(net.YVarTypeLogits)
        YVarTypeCrossEntropy = log_softmax(net.YVarTypeLogits) * -YPH.narrow(1, net.outputShape1[0]+net.outputShape2[0], net.outputShape3[0])
        print(YVarTypeCrossEntropy)
        loss3 = YVarTypeCrossEntropy.sum()
        print("Loss3: " + str(loss3)+"\n")

        print(net.YIndelLengthLogits)
        YIndelLengthCrossEntropy = log_softmax(net.YIndelLengthLogits) * -YPH.narrow(1, net.outputShape1[0]+net.outputShape2[0]+net.outputShape3[0], net.outputShape4[0])
        print(YIndelLengthCrossEntropy)
        loss4 = YIndelLengthCrossEntropy.sum()
        print("Loss4: " + str(loss4)+"\n")

        l2_reg = None
        for name, W in net.named_parameters():
            if 'bias' not in name:
                print(name)
                print("Weights:\n")
                print(W)
                print("\n")
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
        # print(l2_reg)

        lossL2 = l2_reg * net.l2RegularizationLambdaVal
        print("LossL2: " + str(lossL2)+"\n")

        loss = loss1 + loss2 + loss3 + loss4 + lossL2
        net.loss = loss
        # print(loss)

        return loss

    def setLearningRate(self, learningRate=None):
        if learningRate == None:
            self.learningRateVal = self.learningRateVal * self.learningRateDecay
        else:
            self.learningRateVal = learningRate
        return self.learningRateVal

    # def setL2RegularizationLambda(self, l2RegularizationLambda=None):
    #     if  l2RegularizationLambda == None:
    #         self.l2RegularizationLambdaVal = self.l2RegularizationLambdaVal * self.l2RegularizationLambdaDecay
    #     else:
    #         self.l2RegularizationLambdaVal = l2RegularizationLambda
    #     return self.l2RegularizationLambdaVal
    #
    # def saveParameters(self, fn):
    #     with self.g.as_default():
    #         self.saver = tf.train.Saver()
    #         self.saver.save(self.session, fn)
    #
    # def restoreParameters(self, fn):
    #     with self.g.as_default():
    #         self.saver = tf.train.Saver()
    #         self.saver.restore(self.session, fn)

    def predict(self, XArray):
        XArray = torch.from_numpy(XArray).permute(0,3,1,2)
        base, zygosity, varType, indelLength = self.forward(XArray)
        return base, zygosity, varType, indelLength

    def predictNoRT(self, XArray):
        XArray = torch.from_numpy(XArray).permute(0,3,1,2)
        # print(XArray.shape)
        self.predictBaseRTVal = None; self.predictZygosityRTVal = None; self.predictVarTypeRTVal = None; self.predictIndelLengthRTVal = None
        self.predictBaseRTVal, self.predictZygosityRTVal, self.predictVarTypeRTVal, self.predictIndelLengthRTVal = self.forward(XArray)
        # print(self.predictBaseRTVal, self.predictZygosityRTVal, self.predictVarTypeRTVal, self.predictIndelLengthRTVal)

    def train(self, batchX, batchY):
        batchX = torch.from_numpy(batchX).permute(0,3,1,2)
        self.optimizer.zero_grad()
        # print(batchX)
        # print("\n")
        out = self(batchX)
        # print(out)
        # print("\n")
        # Why is loss negative?
        loss = self.costFunction(torch.from_numpy(batchY))
        loss.backward()
        self.optimizer.step()

        return loss, None


    # def train(self, batchX, batchY):
    #     # create your optimizer
    #     optimizer = optim.Adam(net.parameters(), lr=net.learningRateVal)
    #
    #     for epoch in range(10):
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         print(batchX)
    #         print("\n")
    #         out = net(batchX)
    #         print(out)
    #         print("\n")
    #         # Why is loss negative?
    #         loss = self.costFunction(batchY)
    #         loss.backward()
    #         optimizer.step()
    #         print("Epoch: " + str(epoch) + " ----------------------- Loss: " + str(loss) + "\n")
    #
    #     return loss

# if __name__ == "__main__":
#     net = Net()
#     print(net)
#
#     params = list(net.parameters())
#
#     input = torch.randn(1, param.matrixNum, 2*param.flankingBaseNum+1, 4).random_(0,5)
#     YPH = torch.randn(1, 16)
#
#     net.train(input, YPH)
