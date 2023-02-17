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

```sh
conda create -n DPM-GSP python=3.10
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install tqdm
pip install pyyaml
pip install easydict
pip install torch-sparse
pip install torch-scatter==2.0.9
```
You also need to install torch-geometric package. Each experiment requires a different version.

### DPM-GSP for fully-supervised and reasoning  
```sh
pip install torch-geometric==1.7.1
```

### DPM-GSP-semi-supervised  
```sh
pip install torch-geometric==2.1.0
```

## Running
At each directory, you can use the following command to obtain the results of our paper.

```sh
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
--config config_name
```
