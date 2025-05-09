# Deep Learning Advances Arctic River Temperature Predictions

[![DOI](https://zenodo.org/badge/957623666.svg)](https://doi.org/10.5281/zenodo.15186293)

This repository contains the data and code associated with the manuscript titled **"Deep Learning Advances Arctic River Temperature Predictions"**, currently in revision with *Water Resources Research*.

We developed a high-performance Long Short-Term Memory (LSTM) model to improve predictions of river water temperature dynamics in ungauged basins across Alaska. 

Using explainable AI (XAI) techniques, we: (1) verified (to build trust) that the model’s learned patterns align with known physical processes, (2) identified the dominant physical processes correlated with river temperature dynamics in Alaskan systems, and (3) analyzed how the LSTM model learns these dynamics.

## Prerequisite

The LSTM framework was implemented using [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) (Kratzert et al., 2022), an open-source Python library based on PyTorch.

#### Set up

Refer to the [NeuralHydrology Prerequists](https://github.com/neuralhydrology/neuralhydrology/tree/master/environments) to set up a new environment and install necessary dependencies. Choose the appropriate YAML file for your system. If you do have a CUDA-capable GPU, please use the *environment_cuda11_8.yml*. Otherwise, you can do the *environment_cpu.yml*


#### Installation

Refer to the [NeuralHydrology Installation](https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html#installation) to install an editable version of NeuralHydrology. 
The version used for implementation here is v1.9.0.

## Model and Data

The input data to the LSTM can be found in `Data/1_test`. With the model set up defined by `Data/Hyperparameter_18.yml`, we developed an Alaska river water temperature LSTM model (`Scripts/LSTM.py`) and used integrated gradients (`Scripts/Integral_gradients.py`) to better understand it. The results can be found in `runs`. Below are detailed data structure and scripts.

<img src="./Structure.png" style="zoom:60%;" />

#### Data

`1_test`: Processed time-series (.nc) and static attributes (.csv) for NeuralHydrology LSTM 

`Rawdata`: Time-series (.csv) and static attributes (.csv) were obtained from the Veins of the Earth (VotE) platform (Schwenk et al., 2021)

`Hyperparameter_18.yml`: The configuration file used to set up the LSTM (such as train/test split, LSTM hyperparameters, data path and etc). For the meaning of each arguments, please referred to the original [NeuralHydrology](**https://neuralhydrology.readthedocs.io/en/latest/usage/config.html**).

#### Scripts

`LSTM.py`: LSTM model set up, calibration, and evaluation

`LSTM_plot.py`: Analysis and plotting based on LSTM simulations

`Integral_gradients.py`: Calculate the IG. The dimension is 𝐼𝑛𝑡𝑒𝑔𝑟𝑎𝑡𝑒𝑑𝐺𝑟𝑎𝑑𝑠𝑓×𝜏×𝑏×𝑑 , where 𝑓 indicates the number of forcings, 𝜏 is the length of look-back window, 𝑏 is the number of basins, and 𝑑 is the number of dates we conducted integrated gradients analysis on. 

`Integral_gradients_plot.py`: Analysis and plotting based on IG.

#### Run

`Final_20230921_2909_103637`: The results folder stores calibrated LSTM and model evaluation matrix



