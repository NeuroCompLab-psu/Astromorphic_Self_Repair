# Astromorphic_Self_Repair

This repository is prepared for the paper *Astromorphic Self-Repair of Neuromorphic Hardware Systems*.

This paper includes a MATLAB simulation for the dynamics of self-repair interation between neurons and their neighboring astrocyte.
Also, this paper discusses the proposed Spike-Timing-Dependent Plasticity (STDP) learning rule for repairing a degenerated crossbar array system. The STDP network is based on [BindsNET](https://github.com/BindsNET/bindsnet).

---

For the MATLAB simulation (Path: /MATLAB/), there are 3 scripts. The **PR_simulation_stuck_to_zero.m** is a function called by **PR_repair_statistics_parameter_sweep_base_healthy_sum.m** which is a simulation plotting the relationship between the fault severity and the synapse weight gain in self-repair. The file **self_repair_single_simulation_stuck_to_zero.m** shows how the synapse weights of a group of neurons change vs time, under the repair of the neighboring astrocyte.

In summary, **PR_repair_statistics_parameter_sweep_base_healthy_sum.m** and **self_repair_single_simulation_stuck_to_zero.m** are runnable scripts, which could plot the graphs shown in the paper.

---

For the STDP network self-repair (Path: /BindsNET/), several packages are needed, and listed below:

- Python 3.8
- Pytorch
- BindsNET
- Matplotlib
- seaborn


The only runnable script: **DR_WSSTDP_stktz.py**

**cusmodels.py**, **custopology.py** and **cuslearning.py** must be in the same dir as the main script.

bash template:

**Fashion MNIST Dataset:**
```
#!/bin/bash

python ./DR_WSSTDP_stktz.py \
--log-dir ./logs/dir_name \
--FMNIST \
--seed 1 \
--n-neurons 400 \
--time 100 \
--batch-size 16 \
--prob 0.9 \
--weight-max 1000 \
--t-norm 1e4 \
--v-mean 1 \
--v-sigma 0.2258 \
--tau 0.004 \
--kdiv 0 \
--bdiv 2 \
--eq-step 1 \
--sum-lowerbound 0.22 \
--intensity 45 \
--inh 250 \
--learning-rule WeightSumDependentDivergencePostPre \
--nu-pre 4e-5 \
--nu-post 4e-3 \
--sobel \
--update-steps 250 \
--n-epochs 2 \
--test-batch-size 10000 \
--test-time 100 \
--gpu \
--network-file-save \
--network-file-load ./HealthyFashionMNISTNetwork.pt
```


**MNSIT Dataset:**
```
#!/bin/bash

python ./DR_WSSTDP_stktz.py \
--log-dir ./logs/dir_name \
--seed 1 \
--n-neurons 400 \
--time 100 \
--batch-size 16 \
--prob 0.9 \
--weight-max 1000 \
--t-norm 1e4 \
--v-mean 1 \
--v-sigma 0.2258 \
--tau 0.01 \
--kdiv 0 \
--bdiv 2 \
--eq-step 1 \
--sum-lowerbound 0.17 \
--update-steps 250 \
--n-epochs 2 \
--test-batch-size 10000 \
--test-time 100 \
--gpu \
--network-file-save \
--network-file-load ./HealthyMNISTNetwork.pt
```

### Parameter interpretation:

--seed: random number generator seed \
--prob: p_fault \
--kdiv, --bdiv: weight divergence, not included in this paper, for test purpose \
--eq-step: do one normalization per (eq-step) batches \
--update-steps: test and record accuracy per (update-steps) batches \
--network-file-load: path of healthy network saved by using BindsNET's network.save() \


