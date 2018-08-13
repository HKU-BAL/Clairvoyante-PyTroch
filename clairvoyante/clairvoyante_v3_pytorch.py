# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import param
import sys
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

        print(inputShape)

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

        # Used for epoch counting
        self.counter = 1

        # # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.cuda.device(0)
        print(torch.cuda.get_device_name(0))
        # self.to(self.deivce)

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
        print("Shape")
        print(XPH.shape)
        XPH = XPH

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
        print("ZFC: "+str(YZygosityFC)+"\n")
        YZygosityLogits = torch.add(YZygosityFC, epsilon)
        self.YZygosityLogits = YZygosityLogits
        # print(YZygosityLogits)
        YZygositySoftmax = softmax(YZygosityLogits)
        self.YZygositySoftmax = YZygositySoftmax
        # print(YZygositySoftmax.shape)

        YVarTypeFC = selu(self.YVarTypeFCLayer(dropout5))
        print("VFC: "+str(YVarTypeFC)+"\n")
        YVarTypeLogits = torch.add(YVarTypeFC, epsilon)
        self.YVarTypeLogits = YVarTypeLogits
        # print(YVarTypeLogits)
        YVarTypeSoftmax = softmax(YVarTypeLogits)
        self.YVarTypeSoftmax = YVarTypeSoftmax
        # print(YVarTypeSoftmax.shape)

        YIndelLengthFC = selu(self.YIndelLengthFCLayer(dropout5))
        print("IFC: "+str(YIndelLengthFC)+"\n")
        YIndelLengthLogits = torch.add(YIndelLengthFC, epsilon)
        self.YIndelLengthLogits = YIndelLengthLogits
        # print(YIndelLengthLogits)
        YIndelLengthSoftmax = softmax(YIndelLengthLogits)
        self.YIndelLengthSoftmax = YIndelLengthSoftmax
        # print(YIndelLengthSoftmax.shape)

        return YBaseChangeSigmoid.cpu().data.numpy(),YZygositySoftmax.cpu().data.numpy(),YVarTypeSoftmax.cpu().data.numpy(),YIndelLengthSoftmax.cpu().data.numpy()

    def costFunction(self, YPH):
        YPH = YPH.float()
        print("YPH: "+ str(YPH[0]) + "\n")
        # Calculates MSE without computing average.
        mse = nn.MSELoss(reduction='sum')
        loss1 = mse(self.YBaseChangeSigmoid, YPH.narrow(1, 0, self.outputShape1[0]))
        print(self.YBaseChangeSigmoid)
        print(YPH.narrow(1, 0, self.outputShape1[0]))
        print("Loss1: "+str(loss1)+"\n")

        log_softmax = nn.LogSoftmax(dim=1)

        print(self.YZygosityLogits)
        YZygosityCrossEntropy = log_softmax(self.YZygosityLogits) * -YPH.narrow(1, self.outputShape1[0], self.outputShape2[0])
        print(YZygosityCrossEntropy)
        print(YPH.narrow(1, self.outputShape1[0], self.outputShape2[0]))
        loss2 = YZygosityCrossEntropy.sum()
        print("Loss2: "+str(loss2)+"\n")

        print(self.YVarTypeLogits)
        YVarTypeCrossEntropy = log_softmax(self.YVarTypeLogits) * -YPH.narrow(1, self.outputShape1[0]+self.outputShape2[0], self.outputShape3[0])
        print(YVarTypeCrossEntropy)
        print(YPH.narrow(1, self.outputShape1[0]+self.outputShape2[0], self.outputShape3[0]))
        loss3 = YVarTypeCrossEntropy.sum()
        print("Loss3: " + str(loss3)+"\n")

        print(self.YIndelLengthLogits)
        YIndelLengthCrossEntropy = log_softmax(self.YIndelLengthLogits) * -YPH.narrow(1, self.outputShape1[0]+self.outputShape2[0]+self.outputShape3[0], self.outputShape4[0])
        print(YIndelLengthCrossEntropy)
        print(YPH.narrow(1, self.outputShape1[0]+self.outputShape2[0]+self.outputShape3[0], self.outputShape4[0]))
        loss4 = YIndelLengthCrossEntropy.sum()
        print("Loss4: " + str(loss4)+"\n")

        l2_reg = None
        for name, W in self.named_parameters():
            if 'bias' not in name:
                # W = W.to(self.device)
                # print(name)
                # print("Weights:\n")
                # print(W)
                # print("\n")
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
        # print(l2_reg)

        lossL2 = l2_reg * self.l2RegularizationLambdaVal
        print("LossL2: " + str(lossL2)+"\n")

        loss = loss1 + loss2 + loss3 + loss4 + lossL2
        self.loss = loss
        # print("Final Loss" + str(loss))

        return loss

    def setLearningRate(self, learningRate=None):
        if learningRate == None:
            self.learningRateVal = self.learningRateVal * self.learningRateDecay
        else:
            self.learningRateVal = learningRate
        return self.learningRateVal

    def setL2RegularizationLambda(self, l2RegularizationLambda=None):
        if  l2RegularizationLambda == None:
            self.l2RegularizationLambdaVal = self.l2RegularizationLambdaVal * self.l2RegularizationLambdaDecay
        else:
            self.l2RegularizationLambdaVal = l2RegularizationLambda
        return self.l2RegularizationLambdaVal

    def getLoss(self, batchX, batchY):
        batchX = torch.from_numpy(batchX).to(self.device).permute(0,3,1,2)

        out = self(batchX)

        loss = self.costFunction(torch.from_numpy(batchY).to(self.device))

        return loss.data.numpy()

    def getLossNoRT(self, batchX, batchY):

        self.getLossLossRTVal = None

        batchX = torch.from_numpy(batchX).to(self.device).permute(0,3,1,2)

        out = self(batchX)

        loss = self.costFunction(torch.from_numpy(batchY).to(self.device))

        self.getLossLossRTVal = loss.data.numpy()

    def restoreParameters(self, path):
        self.load_state_dict(torch.load(path))
        # f = open("../illumina_parameters.txt", "a")
        # for name, W in self.named_parameters():
        #     f.write(name)
        #     f.write(str(W.shape))
        #     f.write(str(W))
        #     f.write("\n")

    def predict(self, XArray):
        print(XArray.shape)
        print(XArray[0])
        XArray = torch.from_numpy(XArray).to(self.device).permute(0,3,1,2)
        print(XArray[0])
        base, zygosity, varType, indelLength = self.forward(XArray)
        return base, zygosity, varType, indelLength

    def predictNoRT(self, XArray):
        print(XArray.shape)
        print(XArray[0])
        XArray = torch.from_numpy(XArray).to(self.device).permute(0,3,1,2)
        print(XArray[0])
        # print(XArray.shape)
        self.predictBaseRTVal = None; self.predictZygosityRTVal = None; self.predictVarTypeRTVal = None; self.predictIndelLengthRTVal = None
        self.predictBaseRTVal, self.predictZygosityRTVal, self.predictVarTypeRTVal, self.predictIndelLengthRTVal = self.forward(XArray)
        # print(self.predictBaseRTVal, self.predictZygosityRTVal, self.predictVarTypeRTVal, self.predictIndelLengthRTVal)

    def train(self, batchX, batchY):
        batchX = torch.from_numpy(batchX).to(self.device).permute(0,3,1,2)
        self.optimizer.zero_grad()
        # print("BatchX: " + str(batchX[0]))
        # print("\n")
        out = self(batchX)
        # print("Out: " + str(out))
        # print("\n")
        # Why is loss negative?
        loss = self.costFunction(torch.from_numpy(batchY).to(self.device))
        loss.backward()
        self.optimizer.step()

        print("Epoch: " + str(self.counter) + " ---------------------------- Loss: " + str(loss.data.numpy()) + "\n")
        self.counter += 1
        torch.save(self.state_dict(), "../pytorchModels/demoRunCPU_v4Train/demoRunCPU4_parameters.txt")
        sys.stdout.flush()
        return loss.data.numpy(), None

    def trainNoRT(self, batchX, batchY):

        self.trainLossRTVal = None
        self.trainSummaryRTVal = None

        batchX = torch.from_numpy(batchX).to(self.device).permute(0,3,1,2)
        self.optimizer.zero_grad()
        # print("BatchX: " + str(batchX[0]))
        # print("\n")
        out = self(batchX)
        # print("Out: " + str(out))
        # print("\n")
        # Why is loss negative?
        loss = self.costFunction(torch.from_numpy(batchY).to(self.device))
        loss.backward()
        self.optimizer.step()

        print("Epoch: " + str(self.counter) + " ---------------------------- Loss: " + str(loss.data.numpy()) + "\n")
        self.counter += 1
        torch.save(self.state_dict(), "../pytorchModels/trainAll/trainAll_parameters.txt")
        sys.stdout.flush()

        self.trainLossRTVal = loss.data.numpy()

    # def train(self, batchX, batchY):
    #     # create your optimizer
    #     optimizer = optim.Adam(self.parameters(), lr=self.learningRateVal)
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
#     params = list(self.parameters())
#
#     input = torch.randn(1, param.matrixNum, 2*param.flankingBaseNum+1, 4).random_(0,5)
#     YPH = torch.randn(1, 16)
#
#     self.train(input, YPH)
