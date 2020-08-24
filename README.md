# Tensorflow Latent Tensor Reconstruction

Tensorflow 2 implementation of latent tensor reconstruction (LTR). Closely follows the algorithm described by Szedmak et al. in [this paper](https://arxiv.org/abs/2005.01538).

features:
* it is simple to swap out different loss functions and optimizers
* support for dense and sparse inputs
* arbitrary number of ranks can be trained simultaneously
* mini-batch training for memory-efficiency
* track loss and accuracy with Tensorboard
* can be used for classification or ranking with the addition of a suitable activation function

## Installation

I recommend using a clean Python virtual environment, especially if using NVIDIA GPU drivers. Detailed instructions for installing Tensorflow with pip can be found [here](https://www.tensorflow.org/install/pip). In order to add GPU support, please refer to the instructions [here](https://www.tensorflow.org/install/gpu).

Other dependencies:
* [scipy](https://www.scipy.org/install.html)
* [numpy](https://numpy.org/install/)
* [tqdm](https://github.com/tqdm/tqdm#installation)

## Getting started

For a quick introduction on how to use the implementation, please take a look at the example notebook included in this repository.
