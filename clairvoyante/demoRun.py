from __future__ import division
from __future__ import absolute_import
import sys
import time
import argparse
import param
import logging
import numpy as np
import utils_v2 as utils
# import clairvoyante_v2 as cv
import clairvoyante_v3_pytorch as cv
from itertools import izip

logging.basicConfig(format=u'%(message)s', level=logging.INFO)

def Run(args):
    # create a Clairvoyante
    logging.info(u"Initializing model ...")
    utils.SetupEnv()
    m = cv.Net()

    TrainAll(args, m)
    Test22(args, m)

def TrainAll(args, m):
    logging.info(u"Loading the training dataset ...")
    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
    utils.GetTrainingArray(u"../training/tensor_can_chr21",
                           u"../training/var_chr21",
                           u"../training/bed")

    logging.info(u"The size of training dataset: {}".format(total))

    # op to write logs to Tensorboard
    # if args.olog != None:
    #     summaryWriter = m.summaryFileWriter(args.olog)

    # training and save the parameters, we train on the first 90% variant sites and validate on the last 10% variant sites
    logging.info(u"Start training ...")
    trainingStart = time.time()
    trainBatchSize = param.trainBatchSize
    validationLosts = []
    logging.info(u"Start at learning rate: %.2e" % m.setLearningRate(args.learning_rate))
    c = 0; maxLearningRateSwitch = param.maxLearningRateSwitch
    epochStart = time.time()
    datasetPtr = 0
    trainingTotal = int(total*param.trainingDatasetPercentage)
    validationStart = trainingTotal + 1
    numValItems = total - validationStart
    valXArray, _, _ = utils.DecompressArray(XArrayCompressed, validationStart, numValItems, total)
    valYArray, _, _ = utils.DecompressArray(YArrayCompressed, validationStart, numValItems, total)
    logging.info(u"Number of variants for validation: %d" % len(valXArray))
    i = 1
    while i < (1 + int(param.maxEpoch * trainingTotal / trainBatchSize + 0.499)):
        XBatch, num, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, trainBatchSize, trainingTotal)
        YBatch, num2, endFlag2 = utils.DecompressArray(YArrayCompressed, datasetPtr, trainBatchSize, trainingTotal)
        if num != num2 or endFlag != endFlag2:
            sys.exit(u"Inconsistency between decompressed arrays: %d/%d" % (num, num2))
        loss, summary = m.train(XBatch, YBatch)
        # if args.olog != None:
        #     summaryWriter.add_summary(summary, i)
        if endFlag != 0:
            validationLost = m.getLoss( valXArray, valYArray )
            logging.info(u" ".join([unicode(i), u"Training loss:", unicode(loss/trainBatchSize), u"Validation loss: ", unicode(validationLost/numValItems)]))
            logging.info(u"Epoch time elapsed: %.2f s" % (time.time() - epochStart))
            validationLosts.append( (validationLost, i) )
            c += 1
            flag = 0
            flipFlop = 0
            if c >= 6:
              if validationLosts[-6][0] - validationLosts[-5][0] <= 0: flipFlop += 1
              if validationLosts[-5][0] - validationLosts[-4][0] <= 0: flipFlop += 1
              if validationLosts[-4][0] - validationLosts[-3][0] <= 0: flipFlop += 1
              if validationLosts[-3][0] - validationLosts[-2][0] <= 0: flipFlop += 1
              if validationLosts[-2][0] - validationLosts[-1][0] <= 0: flipFlop += 1
            if flipFlop >= 3:
                maxLearningRateSwitch -= 1
                if maxLearningRateSwitch == 0:
                  break
                logging.info(u"New learning rate: %.2e" % m.setLearningRate())
                c = 0
            epochStart = time.time()
            datasetPtr = 0
        i += 1
        datasetPtr += trainBatchSize

    logging.info(u"Training time elapsed: %.2f s" % (time.time() - trainingStart))

    # show the parameter set with the smallest validation loss
    validationLosts.sort()
    i = validationLosts[0][1]
    logging.info(u"Best validation loss at batch: %d" % i)

    logging.info(u"Testing on the training dataset ...")
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    datasetPtr = 0
    XBatch, _, _ = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
    bases = []; zs = []; ts = []; ls = []
    base, z, t, l = m.predict(XBatch)
    bases.append(base); zs.append(z); ts.append(t); ls.append(l)
    datasetPtr += predictBatchSize
    while datasetPtr < total:
        XBatch, _, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
        base, z, t, l = m.predict(XBatch)
        bases.append(base); zs.append(z); ts.append(t); ls.append(l)
        datasetPtr += predictBatchSize
        if endFlag != 0:
            break
    bases = np.concatenate(bases[:]); zs = np.concatenate(zs[:]); ts = np.concatenate(ts[:]); ls = np.concatenate(ls[:])
    print >>sys.stderr, u"Prediciton time elapsed: %.2f s" % (time.time() - predictStart)

    # Evaluate the trained model
    YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
    print >>sys.stderr, u"Version 2 model, evaluation on base change:"
    allBaseCount = top1Count = top2Count = 0
    for predictV, annotateV in izip(bases, YArray[:,0:4]):
        allBaseCount += 1
        sortPredictV = predictV.argsort()[::-1]
        if np.argmax(annotateV) == sortPredictV[0]: top1Count += 1; top2Count += 1
        elif np.argmax(annotateV) == sortPredictV[1]: top2Count += 1
    print >>sys.stderr, u"all/top1/top2/top1p/top2p: %d/%d/%d/%.2f/%.2f" %\
                (allBaseCount, top1Count, top2Count, float(top1Count)/allBaseCount*100, float(top2Count)/allBaseCount*100)
    print >>sys.stderr, u"Version 2 model, evaluation on Zygosity:"
    ed = np.zeros( (2,2), dtype=np.int )
    for predictV, annotateV in izip(zs, YArray[:,4:6]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
    for i in xrange(2):
        print >>sys.stderr, u"\t".join([unicode(ed[i][j]) for j in xrange(2)])
    print >>sys.stderr, u"Version 2 model, evaluation on variant type:"
    ed = np.zeros( (4,4), dtype=np.int )
    for predictV, annotateV in izip(ts, YArray[:,6:10]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
    for i in xrange(4):
        print >>sys.stderr, u"\t".join([unicode(ed[i][j]) for j in xrange(4)])
    print >>sys.stderr, u"Version 2 model, evaluation on indel length:"
    ed = np.zeros( (6,6), dtype=np.int )
    for predictV, annotateV in izip(ls, YArray[:,10:16]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
    for i in xrange(6):
        print >>sys.stderr, u"\t".join([unicode(ed[i][j]) for j in xrange(6)])

def Test22(args, m):
    logging.info(u"Loading the chr22 dataset ...")
    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
    utils.GetTrainingArray(u"../training/tensor_can_chr22",
                           u"../training/var_chr22",
                           u"../training/bed")

    logging.info(u"Testing on the chr22 dataset ...")
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    datasetPtr = 0
    XBatch, _, _ = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
    bases = []; zs = []; ts = []; ls = []
    base, z, t, l = m.predict(XBatch)
    bases.append(base); zs.append(z); ts.append(t); ls.append(l)
    datasetPtr += predictBatchSize
    while datasetPtr < total:
        XBatch, _, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
        base, z, t, l = m.predict(XBatch)
        bases.append(base); zs.append(z); ts.append(t); ls.append(l)
        datasetPtr += predictBatchSize
        if endFlag != 0:
            break
    bases = np.concatenate(bases[:]); zs = np.concatenate(zs[:]); ts = np.concatenate(ts[:]); ls = np.concatenate(ls[:])
    print >>sys.stderr, u"Prediciton time elapsed: %.2f s" % (time.time() - predictStart)

    # Evaluate the trained model
    YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
    print >>sys.stderr, u"Version 2 model, evaluation on base change:"
    allBaseCount = top1Count = top2Count = 0
    for predictV, annotateV in izip(bases, YArray[:,0:4]):
        allBaseCount += 1
        sortPredictV = predictV.argsort()[::-1]
        if np.argmax(annotateV) == sortPredictV[0]: top1Count += 1; top2Count += 1
        elif np.argmax(annotateV) == sortPredictV[1]: top2Count += 1
    print >>sys.stderr, u"all/top1/top2/top1p/top2p: %d/%d/%d/%.2f/%.2f" %\
                (allBaseCount, top1Count, top2Count, float(top1Count)/allBaseCount*100, float(top2Count)/allBaseCount*100)
    print >>sys.stderr, u"Version 2 model, evaluation on Zygosity:"
    ed = np.zeros( (2,2), dtype=np.int )
    for predictV, annotateV in izip(zs, YArray[:,4:6]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
    for i in xrange(2):
        print >>sys.stderr, u"\t".join([unicode(ed[i][j]) for j in xrange(2)])
    print >>sys.stderr, u"Version 2 model, evaluation on variant type:"
    ed = np.zeros( (4,4), dtype=np.int )
    for predictV, annotateV in izip(ts, YArray[:,6:10]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
    for i in xrange(4):
        print >>sys.stderr, u"\t".join([unicode(ed[i][j]) for j in xrange(4)])
    print >>sys.stderr, u"Version 2 model, evaluation on indel length:"
    ed = np.zeros( (6,6), dtype=np.int )
    for predictV, annotateV in izip(ls, YArray[:,10:16]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
    for i in xrange(6):
        print >>sys.stderr, u"\t".join([unicode(ed[i][j]) for j in xrange(6)])


if __name__ == u"__main__":

    parser = argparse.ArgumentParser(
            description=u"Training and testing Clairvoyante using demo dataset" )

    parser.add_argument(u'--learning_rate', type=float, default = param.initialLearningRate,
            help=u"Set the initial learning rate, default: %(default)s")

    parser.add_argument(u'--olog', type=unicode, default = None,
            help=u"Prefix for tensorboard log outputs, optional")

    args = parser.parse_args()

    Run(args)
