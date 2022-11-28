# Astromorphic Self-Repair of Neuromorphic Hardware Systems

## Overview

This codebase contains a simulation of the astrocyte-guided self-repair algorithm of Spike-Timing-Dependent Plasticity (STDP) network running on neuromorphic hardware. The astromorphic STDP self-repair learning rule is extracted from the dynamics of intereaction between a group of neurons and a neighboring astrocyte. When there are faults, for example weight decay or conductance stuck to zero, in the crossbar array system for a STDP network, the network accuracy will degenerate. The proposed algorithm can restore the system accuracy to a reasonable level based on the information remaining in the healthy synapses. Here the STDP network implementation is from 

## Package Requiremnts

- Python >= 3.8
- Pytorch 1.13.0 (with CUDA 11.6 and torchvision)
- BindsNET, Matplotlib and seaborn
- MNIST and Fashion MNIST dataset (included in torchvision)

## How to run a simulation

The healthy weight sets for MNIST and Fashion MNIST datasets are already prepared under path "/BindsNET/" with recognition accuracy correspondingly 90.43% and 77.60%.

Run the script **A_STDP_Self_Repair_main.py** with the following formulas for Fashion MNIST and MNIST separately:

### MNSIT Dataset
```
python ./A_STDP_Self_Repair_main.py
--log-dir
--seed
--n-neurons
--time
--batch-size
--prob
--weight-max
--t-norm
--v-mean
--v-sigma
--tau
--eq-step
--sum-lowerbound
--update-steps
--n-epochs
--test-batch-size
--test-time
--gpu
--network-file-save
--network-file-load
```

### Fashion MNIST

```
python ./A_STDP_Self_Repair_main.py
--log-dir
--FMNIST
--seed
--n-neurons
--time
--batch-size
--prob
--weight-max
--t-norm
--v-mean
--v-sigma
--tau
--eq-step
--sum-lowerbound
--update-steps
--n-epochs
--test-batch-size
--test-time
--gpu
--network-file-save
--network-file-load

--intensity
--inh
--learning-rule
--nu-pre
--nu-post
--sobel
```

### Parameter interpretation and optimal values

Common parameters for Fashion MNSIT and MNIST:
```
--log-dir: Path for simulation log. 
--FMNIST: The script uses FMNIST dataset if this argument presents, and uses MNSIT otherwise. 
--seed: Random number generator seed. It could be any numerical value. 
--n-neurons: Number of neurons in the output layer of the STDP network. 400 is adopted in the paper for both datasets.
--time: The time duration for a single STDP network run in learning mode. 100 is used in the paper.
--batch-size: STDP re-training batch size. 16 is adopted in the paper.
--prob: p_fault, the severity of the faults, tested from 0.5 to 0.9 in the paper. (refer to the paper text for more detail) 
--weight-max: Maximum possible weight for all synapses. Use 1000 or greater values for astrocyte guided self-repair. 
--t-norm: Normalized time in the Phase Change Memory (PCM) conductance decay model. In the paper we use 1e4. 
--v-mean: Mean of v in the Phase Change Memory (PCM) conductance decay model. In the paper we use 1. 
--v-sigma: Standard deviation of v in the Phase Change Memory (PCM) conductance decay model. In the paper we use 0.2258. 
--tau: Self-repair time constant. 0.01 is optimal for MNIST, and 0.004 is optimal for Fashion MNIST. 
--eq-step: After each "eq-step" batches, a normalization is applied. E.g. "--eq-step 1" leads to normalization after each batch of STDP re-training, and "--eq-step 2" leads to normalization after each second batch. 1 is optimal.
--sum-lowerbound: The lower bound of the ratio between remaining sum of weight and original sum of weights for each neuron. If all the neurons in a network are degenerated to a very severe extend, the output spike of a single learning batch will be 0, which will lead to failure of learning for STDP network. So all the neurons will have their sum of wneueight scaled to (sum-lowerbound)*(original sum of weight), if the mean of sum of weight for all the neurons is lower than (sum-lowerbound)*(original sum of weight). 0.17 for MNSIT and 0.22 for Fashion MNIST. 
--update-steps: After each "update-steps" batches, the script measure and record the accuracy of the network, including the weight map of the network. 250 is small enough for parameter tuning. The smaller this value is, the more the time cost. If this value is too large, the highest accuracy will be missed.
--n-epochs: Number of epochs of STDP re-training. 2 epochs is recommanded for both datasets. 
--test-batch-size: 10000 is adopted in the paper. This value should be adjusted based on the GPU memory size. 
--test-time: The time duration for a single STDP network run in inference mode. 100 is used in the paper. 
--gpu: Whether to use GPU for tranining. 
--network-file-save: Same the network weight map each time when the accuracy is measured, if this argument presents. 
--network-file-load: The path to the healthy weight map of a network. A weight mask will be applied to the healthy weight map for generating a faulty weight map, in the script. 
```

Parameters for Fashion MNSIT specifically:
```
--intensity: The input poisson spike train rate to the input image pixel whose brightness is 1. 45 is adopted in the paper. 
--inh: Lateral inhibitory synapse weight. 250 is adopted in the paper. 
--learning-rule: The learning rules used for Fashion MNIST. Always use WeightSumDependentDivergencePostPre. 
--nu-pre: Pre-synaptic learning rate. 4e-5 is adopted in the paper. 
--nu-post: Post-synaptic learning rate. 4e-3 is adopted in the paper. 
--sobel: Whether to use sobel filter to process the input image for edge detection. The script uses sobel filter if this argument presents. 
```



