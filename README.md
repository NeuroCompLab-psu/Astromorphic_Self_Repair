# Astromorphic Self-Repair of Neuromorphic Hardware Systems

## Overview

This codebase contains a simulation of the astrocyte-guided self-repair algorithm of the Spike-Timing-Dependent Plasticity (STDP) network running on neuromorphic hardware. The astromorphic STDP self-repair learning rule is extracted from the dynamics of interaction between a group of neurons and a neighboring astrocyte. When there are faults, for example, weight decay or conductance stuck to zero, in the crossbar array system for an STDP network, the network accuracy will degenerate. The proposed algorithm can restore the system accuracy to a reasonable level based on the information remaining in the healthy synapses. Here the STDP network implementation is from [BindsNET](https://github.com/BindsNET/bindsnet).

## Package Requiremnts

- Python >= 3.8
- Pytorch 1.13.0 (with CUDA 11.6 and torchvision)
- BindsNET, Matplotlib and seaborn
- MNIST and Fashion MNIST dataset (included in torchvision)

## Running a simulation

The healthy weight sets for MNIST (**uncorrupted_MNIST.wt**) and Fashion MNIST (**uncorrupted_Fashion_MNIST.wt**) datasets are already prepared under the path "/BindsNET/".

Run the script **A_STDP_Self_Repair_main.py** with the following formula:

### MNSIT Dataset
```
python ./A_STDP_Self_Repair_main.py \
--network-file-load ./uncorrupted_MNIST.wt \
--dataset MNIST \
--n-neurons 400 \
--time 100 \
--test-time 100 \
--batch-size 16 \
--test-batch-size 10000 \
--inh 120 \
--theta-plus 0.05 \
--tc-theta-decay 1e7 \
--intensity 128 \
--weight-max 1000 \
--nu-pre 1e-4 \
--nu-post 1e-2 \
--thresh -52 \
--pfault 0.8 \
--t-norm 1e4 \
--v-mean 1 \
--v-sigma 0.2258 \
--tau 0.004 \
--eq-step 1 \
--sum-lowerbound 0.17 \
--n-epochs 2 \
--log-dir ./logs/A_STDP_test \
[--gpu] \
[--sobel] \
[--dt 1] \
[--seed 1] \
[--update-steps 250] \
[--network-param-save] \
[--n-workers -1]
```
\* The arguments inside [] are optional.

### Parameter interpretation and default(optimal) values

```
--network-file-load: The path to the healthy weight and theta of a network. A weight mask will be applied to the healthy weight map for generating a faulty weight map, in the script. Example: ./uncorrupted_MNIST.wt
--dataset: Specifying the dataset. This argument can only be "MNIST" or "FMNIST".
--n-neurons: Number of neurons in the output layer of the STDP network. Default: 400(MNIST and Fashion MNIST)
--time: The time duration for a single STDP network run in learning mode. Default: 100
--test-time: The time duration for a single STDP network run in inference mode. Default: 100
--batch-size: STDP batch size for re-training. Default: 16
--test-batch-size: STDP batch size for inference. Default: 16
--inh: Lateral inhibitory synapse weight. Default: 120(MNIST), 250(Fashion MNIST)
--theta-plus: The increment of threshold per slike. Default: 0.05
--tc-theta-decay: the decay time constant of theta. Default: 1e7
--intensity: The input poisson spike train rate to the input image pixel whose brightness is 1. Default: 128(MNIST), 45(Fashion MNIST)
--weight-max: Maximum possible weight for all synapses. Default: 1000
--nu-pre: Pre-synaptic learning rate. Default: 1e-4(MNIST), 4e-5(Fashion MNIST)
--nu-post: Post-synaptic learning rate. Default: 1e-2(MNIST), 4e-3(Fashion MNIST)
--thresh: membrane threshold. Default: -52
--p_fault: The probability of disabling each synapse, tested from 0.5 to 0.9 in the paper. (refer to the paper text for more detail) 
--t-norm: Normalized time in the Phase Change Memory (PCM) conductance decay model. Default: 1e4
--v-mean: Mean of v in the Phase Change Memory (PCM) conductance decay model. Default: 1
--v-sigma: Standard deviation of v in the Phase Change Memory (PCM) conductance decay model. Default: 0.2258
--tau: Self-repair time constant. Default: 0.01(MNIST), 0.004(Fashion MNIST)
--eq-step: After each "eq-step" batches, a normalization is applied. E.g. "--eq-step 1" leads to normalization after each batch of STDP re-training, and "--eq-step 2" leads to normalization after each second batch. Default: 1
--sum-lowerbound: The lower bound of the ratio between remaining sum of weight and original sum of weights for each neuron. If all the neurons in a network are degenerated to a very severe extend, the output spike of a single learning batch will be 0, which will lead to failure of learning for STDP network. So all the neurons will have their sum of wneueight scaled to (sum-lowerbound)*(original sum of weight), if the mean of sum of weight for all the neurons is lower than (sum-lowerbound)*(original sum of weight). Default: 0.17(MNSIT), 0.22(Fashion MNIST)
--n-epochs: Number of epochs of STDP re-training. 2 epochs is recommanded for both datasets. 
--log-dir: Path for simulation log. 
[--gpu]: Whether to use GPU for tranining. 
[--sobel]: Whether to use sobel filter to process the input image for edge detection. The script uses sobel filter if this argument presents. Default: Do not use sobel for MNIST dataset, and use sobel for Fashion MNIST dataset.
[--dt]: Simulation step size. Default: 1
[--seed]: Random number generator seed. It could be any numerical value. 
[--update-steps]: After each "update-steps" batches, the script measure and record the accuracy of the network, including the weight map of the network. The smaller this value is, the more the time cost. If this value is too large, the highest accuracy will be missed. Default: 250
[--network-param-save]: If this argument presents, the script saves the network weight and theta each time when the accuracy is measured.
[--n-workers]: number of workers for dataloaders. This argument cannot be 0. The script uses torch.cuda.device_count() workers if this argument is set to -1 or not present.
```

## Reference
Please cite this code with the following bibliography:

Zhuangyu Han, A. N. M. Nafiul Islam, Abhronil Sengupta, “Astromorphic Self-Repair of Neuromorphic Hardware Systems“, AAAI Conference on Artificial Intelligence (AAAI), 2023

```
@article{https://doi.org/10.48550/arxiv.2209.07428,
  doi = {10.48550/ARXIV.2209.07428},
  url = {https://arxiv.org/abs/2209.07428},
  author = {Han, Zhuangyu and Islam, A N M Nafiul and Sengupta, Abhronil},
  keywords = {Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Astromorphic Self-Repair of Neuromorphic Hardware Systems},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

