# Clairvoyante-pt
Pytorch version of Clairvoyante. 

The main file is `clairvoyante/clairvoyante_v3_pytorch.py` which contains the code for the Pytorch model. It has the exact same APIs as the tensorflow Clairvoyante model in https://github.com/aquaskyline/Clairvoyante/blob/rbDev/clairvoyante/clairvoyante_v3.py. 

The code initialises Clairvoyante with 3 convolutional layers, 2 hidden fully connected layers and 4 output layers. It 
specifies the parameters for these layers and it initialises the network's weights using He initializtion. 

Pytorch uses NCHW format for tensor dimensions so all tensors require permutation in order to be used by the code.

## How to use the module
Initialise the model in the run function in train.py and callVar.py using 
```python
{module name}.Net()
```

Add
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 0:
    m.to(device)
```
to the run function in train.py and callVar.py after initialising the model to use one or more GPUs.

## GPU
Use the `CUDA_VISIBLE_DEVICE` environment variable to specify the GPUs to use. This can be done using the command `export 
CUDA_VISIBLE_DEVICES="$i"`, where `$i` is an integer from 0 identifying the seqeunce of the GPU to be used. The code supports 
GPU parallelism. If no GPUs are specified, the CPU is used instead.

## Folder Stucture and Program Descriptions

`clairvoyante/` | Contains the Pytorch Model  
---|---
`clairvoyante_v3_pytorch.py` | Pytorch Model of Clairvoyante 
`clairvoyante_v3_pytorch_test.py` | Unit test cases to test Pytorch model's loss function 

`correctVCFs/` | Contains the VCFs produced by TF Clairvoyante and training and testing data sets 
---|---
`basic_luo_chr21.vcf` | VCF produced by CallVAr using model produced by `demoRun.py`
`correct_21.vcf	` | `chr21.vcf` in testingData
`luo_bam_21.vcf` | VCF produced by CallVarBam using fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e-3.epoch500
`luo_tensor_can_21.vcf` | VCF produced by CallVar using fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e3.epoch500
`ngmlr1_chr19.vcf` | VCF produced by CallVarBam using fullv3-ont-ngmlr-hg001-hg19

`evalResults/` | Each folder contains a results for a different vcf-eval. The results are at `summary/summary.txt` in each folder. 
---|---
`TrainBamCPU_chr21/` | Comparison between VCF made by `train.py` and `CallVarBam.py` and `correct_21.vcf`. **(Used in presentation)**
`basicLuo_correct/` | Comparison between VCF made by `train.py` and `correct_21.vcf`. **(Used in presentation)**
`correct_bam/` | Comparison between VCF made by fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e3.epoch500 using CallVarBam and `correct_21.vcf`.
`luo_correct/` | Comparison between VCF made by fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e3.epoch500 using CallVar and `correct_21.vcf`.
`ngmlr1_chr19/` | Comparison between VCF made by fullv3-ont-ngmlr-hg001-hg19 using CallVarBam and `/nas7/yswong/base/hg19_chr19.vcf.gz`. **(Used in presentation)**
`trainAll2_chr19/` | Second comparison betwen VCF produced by CallVarBam using fullv3-ont-ngmlr-hg001-hg19 and `/nas7/yswong/base/hg19_chr19.vcf.gz` using GTX 980. **(Used in presentation)**
`trainAll3_chr19/` | Comparison betwen VCF produced by CallVarBam using fullv3-ont-ngmlr-hg001-hg19 and `/nas7/yswong/base/hg19_chr19.vcf.gz` using GTX Titan and GTX 1080 Ti with training batch size at 5000.
`trainAll4_chr19/` | Comparison betwen VCF produced by CallVarBam using fullv3-ont-ngmlr-hg001-hg19 and `/nas7/yswong/base/hg19_chr19.vcf.gz` using GTX Titan and GTX 1080 Ti with training batch size at 10000.
`trainAll_correct/` | Comparison betwen VCF produced by CallVarBam using fullv3-ont-ngmlr-hg001-hg19 and `/nas7/yswong/base/hg19_chr19.vcf.gz` using GTX 980.

`pytorchModels/` | Each folder is a training experiment. Each folder contains the output of each training and some also contains the model parameters stored in a txt file. All models uses `/nas7/yswong/trainingData/tensor_all.bin` to train.
---|---
`trainAll/` | Model produced by training using the GTX 980.
`trainAll2/` | Model produced after training a second time using the GTX 980.
`trainAll3_5000PGPU` | Model produced after training using the GTX 1080 Ti and GTX Titan using a training batch size of 5000.
`trainAll4_10000PGPU` | Model produced after training using the GTX 1080 Ti and GTX Titan using a training batch size of 10000.
`trainAll5_1080Ti` | Output produced after training using the GTX 1080 Ti.
`trainAll6_Titan` | Output produced after training using the GTX Titan.
`trainAll7_2_1080_Ti` | Output produced after training using two GTX 1080 Ti.

