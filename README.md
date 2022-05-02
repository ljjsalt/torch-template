# torch-template

This is a template for any PyTorch projects, it contains code from other open-source projects.



## 1. Prerequisite

- Python
- PyTorch
- [Weights and Biases](https://wandb.ai/site) account for logging experiments.

## 2. Quick Start

```shell
git clone https://github.com/ljjsalt/torch-template # or folk this repository
bash ./setup.sh

# to train a neural network (you have to first add your model to the code, and changes code in main.py)
python train.py --model model_name
CUDA_VISIBLE_DEVICES=1 train.py --model model_name # use a different gpu
```

## 3. Result & Experiment Settings

See Wandb.
<!-- You can find in [wandb](https://wandb.ai/ljjsalt/mlp_mixer?workspace=user-ljjsalt) -->

## 4. Supported Datasets

- CIFAR-10 & CIFAR-100
- SVHN (The Street View House Numbers)
- ImageNet
- TODO

## 4. Resources

...
