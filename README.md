# Diffusion Probabilistic Models for Graph Structured predictions
This is an official implementation of our paper Diffusion Probabilistic Models for Graph Structured predictions.

## Repository Overview
We provide the PyTorch implementation for DPM-GSP framework here. The repository is organised as follows:

```python
|-- DPM-GSP-{fully-supervsed, semi-supervsed, reasoning} # DPM-GSP for supervised node classification, semi-supervised node classification, and reasoning tasks
    |-- config/ # configurations
    |-- parsers/ # the argument parser
    |-- models/ # model definition
    |-- method_series/ # training method
    |-- data/ # dataset
    |-- logs_train/ # training logs
    |-- utils/ # data process and others
    |-- main.py # the training code
```

## Setting up the environment
You can set up the environment by following commands.

### DPM-GSP-fully-supervised  
```sh
conda create --name <env> --file requirements_supervised.txt
```

### DPM-GSP-semi-supervised  
```sh
conda create --name <env> --file requirements_semi-supervised.txt
```

### DPM-GSP-reasoning
```sh
conda create --name <env> --file requirements_supervised.txt
```

## Running
At each directory, you can use the following command to obtain the results of our paper.

```sh
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
--config config_name
```
