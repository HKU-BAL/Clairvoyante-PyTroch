from __future__ import division
from __future__ import absolute_import
import sys
import os
import time
import argparse
import param
import logging
import numpy as np
from threading import Thread
from math import log
import clairvoyante_v3_pytorch as ct
from itertools import izip
from io import open

logging.basicConfig(format=u'%(message)s', level=logging.INFO)
num2base = dict(list(izip((0, 1, 2, 3), u"ACGT")))
v1Type2Name = dict(list(izip((0, 1, 2, 3, 4), (u'HET', u'HOM', u'INS', u'DEL', u'REF'))))
v2Zygosity2Name = dict(list(izip((0, 1), (u'HET', u'HOM'))))
v2Type2Name = dict(list(izip((0, 1, 2, 3), (u'REF', u'SNP', u'INS', u'DEL'))))
v2Length2Name = dict(list(izip((0, 1, 2, 3, 4, 5), (u'0', u'1', u'2', u'3', u'4', u'4+'))))
maxVarLength = 5
inferIndelLengthMinimumAF = 0.125

def Run(args):
    # create a Clairvoyante
    logging.info(u"Loading model ...")
    if args.v2 == True:
        import utils_v2 as utils
        utils.SetupEnv()
        if args.slim == True:
            import clairvoyante_v2_slim as cv
        else:
            import clairvoyante_v2 as cv
    elif args.v3 == True:
        import utils_v2 as utils # v3 network is using v2 utils
        utils.SetupEnv()
        if args.slim == True:
            import clairvoyante_v3_slim as cv
        else:
            import clairvoyante_v3 as cv
    if args.threads == None:
        if args.tensor_fn == u"PIPE":
            param.NUM_THREADS = 4
    else:
        param.NUM_THREADS = args.threads
    m = ct.Net()
    # print(m)
    # m.init()

    # m.restoreParameters(os.path.abspath(args.chkpnt_fn))
    Test(args, m, utils)


def Output(args, call_fh, num, XBatch, posBatch, base, z, t, l):
    if args.v2 == True or args.v3 == True:
        # print base
        if num != len(base):
          sys.exit(u"Inconsistent shape between input tensor and output predictions %d/%d" % (num, len(base)))
        #          --------------  ------  ------------    ------------------
        #          Base chng       Zygo.   Var type        Var length
        #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
        #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
        for j in xrange(len(base)):
            # print base[j]
            # print "\n"
            if args.showRef == False and np.argmax(t[j]) == 0: continue
            # Get variant type, 0:REF, 1:SNP, 2:INS, 3:DEL
            varType = np.argmax(t[j])
            # Get zygosity, 0:HET, 1:HOM
            varZygosity = np.argmax(z[j])
            # Get Indel Length, 0:0, 1:1, 2:2, 3:3, 4:4, 5:>4
            varLength = np.argmax(l[j])
            # Get chromosome, coordination and reference bases with flanking param.flankingBaseNum flanking bases at coordination
            chromosome, coordination, refSeq = posBatch[j].split(u":")
            # Get genotype quality
            sortVarType = np.sort(t[j])[::-1]
            sortZygosity = np.sort(z[j])[::-1]
            sortLength = np.sort(l[j])[::-1]
            qual = int(-4.343 * log((sortVarType[1]*sortZygosity[1]*sortLength[1]  + 1e-300) / (sortVarType[0]*sortZygosity[0]*sortLength[0]  + 1e-300)))
            if qual > 999: qual = 999
            # Get possible alternative bases
            sortBase = base[j].argsort()[::-1]
            base1 = num2base[sortBase[0]]
            base2 = num2base[sortBase[1]]
            # Initialize other variables
            refBase = u""; altBase = u""; inferredIndelLength = 0; dp = 0; info = [];
            # For SNP
            if varType == 1 or varType == 0: # SNP or REF
                coordination = int(coordination)
                refBase = refSeq[param.flankingBaseNum]
                if varType == 1: # SNP
                    altBase = base1 if base1 != refBase else base2
                    #altBase = "%s,%s" % (base1, base2)
                elif varType == 0: # REF
                    altBase = refBase
                dp = sum(XBatch[j,param.flankingBaseNum,:,0] + XBatch[j,param.flankingBaseNum,:,3])
            elif varType == 2: # INS
                # infer the insertion length
                if varLength == 0: varLength = 1
                dp = sum(XBatch[j,param.flankingBaseNum+1,:,0] + XBatch[j,param.flankingBaseNum+1,:,1])
                if varLength != maxVarLength:
                    for k in xrange(param.flankingBaseNum+1, param.flankingBaseNum+varLength+1):
                        altBase += num2base[np.argmax(XBatch[j,k,:,1])]
                else:
                    for k in xrange(param.flankingBaseNum+1, 2*param.flankingBaseNum+1):
                        referenceTensor = XBatch[j,k,:,0]; insertionTensor = XBatch[j,k,:,1]
                        if k < (param.flankingBaseNum + maxVarLength) or sum(insertionTensor) >= (inferIndelLengthMinimumAF * sum(referenceTensor)):
                            inferredIndelLength += 1
                            altBase += num2base[np.argmax(insertionTensor)]
                        else:
                            break
                coordination = int(coordination)
                refBase = refSeq[param.flankingBaseNum]
                # insertions longer than (param.flankingBaseNum-1) are marked SV
                if inferredIndelLength >= param.flankingBaseNum:
                    altBase = u"<INS>"
                    info.append(u"SVTYPE=INS")
                else:
                    altBase = refBase + altBase
            elif varType == 3: # DEL
                if varLength == 0: varLength = 1
                dp = sum(XBatch[j,param.flankingBaseNum+1,:,0] + XBatch[j,param.flankingBaseNum+1,:,2])
                # infer the deletion length
                if varLength == maxVarLength:
                    for k in xrange(param.flankingBaseNum+1, 2*param.flankingBaseNum+1):
                        if k < (param.flankingBaseNum + maxVarLength) or sum(XBatch[j,k,:,2]) >= (inferIndelLengthMinimumAF * sum(XBatch[j,k,:,0])):
                            inferredIndelLength += 1
                        else:
                            break
                # deletions longer than (param.flankingBaseNum-1) are marked SV
                coordination = int(coordination)
                if inferredIndelLength >= param.flankingBaseNum:
                    refBase = refSeq[param.flankingBaseNum]
                    altBase = u"<DEL>"
                    info.append(u"SVTYPE=DEL")
                elif varLength != maxVarLength:
                    refBase = refSeq[param.flankingBaseNum:param.flankingBaseNum+varLength+1]
                    altBase = refSeq[param.flankingBaseNum]
                else:
                    refBase = refSeq[param.flankingBaseNum:param.flankingBaseNum+inferredIndelLength+1]
                    altBase = refSeq[param.flankingBaseNum]
            if inferredIndelLength > 0 and inferredIndelLength < param.flankingBaseNum: info.append(u"LENGUESS=%d" % inferredIndelLength)
            infoStr = u""
            if len(info) == 0: infoStr = u"."
            else: infoStr = u";".join(info)
            gtStr = u""
            if varType == 0: gtStr = u"0/0"
            elif varZygosity == 0: gtStr = u"0/1"
            elif varZygosity == 1: gtStr = u"1/1"

            print >>call_fh, u"%s\t%d\t.\t%s\t%s\t%d\t.\t%s\tGT:GQ:DP\t%s:%d:%d" % (chromosome, coordination, refBase, altBase, qual, infoStr, gtStr, qual, dp)


def PrintVCFHeader(args, call_fh):
    print >>call_fh, u'##fileformat=VCFv4.1'
    print >>call_fh, u'##ALT=<ID=DEL,Description="Deletion">'
    print >>call_fh, u'##ALT=<ID=INS,Description="Insertion of novel sequence">'
    print >>call_fh, u'##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">'
    print >>call_fh, u'##INFO=<ID=LENGUESS,Number=.,Type=Integer,Description="Best guess of the indel length">'
    print >>call_fh, u'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'
    print >>call_fh, u'##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">'
    print >>call_fh, u'##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">'
    print >>call_fh, u'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s' % (args.sampleName)

def Test(args, m, utils):
    call_fh = open(args.call_fn, u"w")
    if args.v2 == True or args.v3 == True:
        PrintVCFHeader(args, call_fh)
    tensorGenerator = utils.GetTensor( args.tensor_fn, param.predictBatchSize )
    # print(tensorGenerator)
    logging.info(u"Calling variants ...")
    predictStart = time.time()
    end = 0; end2 = 0; terminate = 0
    end2, num2, XBatch2, posBatch2 = tensorGenerator.next()
    # print(XBatch2.shape)
    m.predictNoRT(XBatch2)
    base = m.predictBaseRTVal; z = m.predictZygosityRTVal; t = m.predictVarTypeRTVal; l = m.predictIndelLengthRTVal
    # print(base.shape, z.shape, t.shape, l.shape)
    if end2 == 0:
        end = end2; num = num2; XBatch = XBatch2; posBatch = posBatch2
        end2, num2, XBatch2, posBatch2 = tensorGenerator.next()
        while True:
            if end == 1:
                terminate = 1
            threadPool = []
            if end == 0:
                threadPool.append(Thread(target=m.predictNoRT, args=(XBatch2, )))
            threadPool.append(Thread(target=Output, args=(args, call_fh, num, XBatch, posBatch, base, z, t, l, )))
            for t in threadPool: t.start()
            if end2 == 0:
                end3, num3, XBatch3, posBatch3 = tensorGenerator.next()
            for t in threadPool: t.join()
            base = m.predictBaseRTVal; z = m.predictZygosityRTVal; t = m.predictVarTypeRTVal; l = m.predictIndelLengthRTVal
            if end == 0:
                end = end2; num = num2; XBatch = XBatch2; posBatch = posBatch2
            if end2 == 0:
                end2 = end3; num2 = num3; XBatch2 = XBatch3; posBatch2 = posBatch3
            #print >> sys.stderr, end, end2, end3, terminate
            if terminate == 1:
                break
    elif end2 == 1:
        Output(args, call_fh, num2, XBatch2, posBatch2, base, z, t, l)

    logging.info(u"Total time elapsed: %.2f s" % (time.time() - predictStart))


if __name__ == u"__main__":

    parser = argparse.ArgumentParser(
            description=u"Call variants using a trained Clairvoyante model and tensors of candididate variants" )

    parser.add_argument(u'--tensor_fn', type=unicode, default = u"PIPE",
            help=u"Tensor input, use PIPE for standard input")

    parser.add_argument(u'--chkpnt_fn', type=unicode, default = None,
            help=u"Input a checkpoint for testing or continue training")

    parser.add_argument(u'--call_fn', type=unicode, default = None,
            help=u"Output variant predictions")

    parser.add_argument(u'--sampleName', type=unicode, default = u"SAMPLE",
            help=u"Define the sample name to be shown in the VCF file")

    parser.add_argument(u'--showRef', type=param.str2bool, nargs=u'?', const=True, default = False,
            help=u"Show reference calls, optional")

    parser.add_argument(u'--threads', type=int, default = None,
            help=u"Number of threads, optional")

    parser.add_argument(u'--v3', type=param.str2bool, nargs=u'?', const=True, default = True,
            help=u"Use Clairvoyante version 3")

    parser.add_argument(u'--v2', type=param.str2bool, nargs=u'?', const=True, default = False,
            help=u"Use Clairvoyante version 2")

    parser.add_argument(u'--slim', type=param.str2bool, nargs=u'?', const=True, default = False,
            help=u"Train using the slim version of Clairvoyante, optional")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
