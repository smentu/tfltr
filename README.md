# Tensorflow Latent Tensor Reconstruction

This is a Tensorflow 2 implementation of latent tensor reconstruction (LTR). LTR can be seen as a modification to factorization machines (FM) that removes some of the restrictions on the latent representation, allowing for greater model flexibility, with only a modest increase in computational cost. For an introduction to latent tensor models, please refer to the [original LTR paper](https://arxiv.org/abs/2005.01538) or my master's thesis that is included in this repository. This implementation closely follows the algorithm described by Szedmak et al.

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
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [numpy](https://numpy.org/install/)
* [tqdm](https://github.com/tqdm/tqdm#installation)

## Getting started

For a quick introduction on how to use the implementation, please take a look at the example notebook included in this repository. 
