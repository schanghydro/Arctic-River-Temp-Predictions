# Deep Learning Advances Arctic River Temperature Predictions:snowflake::mount_fuji:

This repository contains the data and code associated with the manuscript titled **"Deep Learning Advances Arctic River Temperature Predictions"**, currently in revision with *Water Resources Research*.

We developed a high-performance Long Short-Term Memory (LSTM) model to improve predictions of river water temperature dynamics in ungauged basins across Alaska. 

Using explainable AI (XAI) techniques, we: (1) verified (to build trust) that the model’s learned patterns align with known physical processes, (2) identified the dominant physical processes correlated with river temperature dynamics in Alaskan systems, and (3) analyzed how the LSTM model learns these dynamics.

## Model

The LSTM framework was implemented using [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) (Kratzert et al., 2022), an open-source Python library based on PyTorch.

#### Set up

Refer to the [NeuralHydrology Prerequists](https://github.com/neuralhydrology/neuralhydrology/tree/master/environments) to set up a new environment. Choose the appropriate YAML file for your system. 
If you do have a CUDA-capable GPU, please use the *environment_cuda11_8.yml*. Otherwise, you can do the *environment_cpu.yml*


#### Installation

Refer to the [NeuralHydrology Installation](https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html#installation) to install an editable version of NeuralHydrology 
The version used for implementation here is v1.9.0.

## Structure

#### 



