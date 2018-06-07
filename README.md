# q-Space Novelty Detection with Variational Autoencoders
This repository contains the official implementation for the paper [q-Space Novelty Detection with Variational Autoencoders]().

# Dependencies:
* python 3
* theano
* lasagne
* numpy
* scipy
* matlab

# Methods:
Distance- and density-based methods are implemented in matlab. Other methods are implemented in python.

# Usage:
To use one of the proposed novelty detection methods with your data you should:
1. implement methods in model/Data.py
2. train a model on your data
3. run one of the proposed methods (for matlab methods, data should be saved in 'mat' format first)