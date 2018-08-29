# Clairvoyante-pt
Pytorch version of Clairvoyante. 

The main file is `clairvoyante_v3_pytorch.py` which contains the code for the Pytorch model. It has the exact same APIs as the tensorflow Clairvoyante model in https://github.com/aquaskyline/Clairvoyante/blob/rbDev/clairvoyante/clairvoyante_v3.py. 

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
to the run function in train.py and callVar.py after initialising the model to use one or more GPU.

## GPU
Use the `CUDA_VISIBLE_DEVICE` environment variable to specify the GPUs to use. This can be done using the command `export 
CUDA_VISIBLE_DEVICES="$i"`, where `$i` is an integer from 0 identifying the seqeunce of the GPU to be used. The code supports 
GPU parallelism. If no GPUs are specified, the CPU is used instead.

