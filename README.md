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
1. implement data loading methods in model/Data.py
2. train a model on your data
3. run one of the proposed methods:
```python
# test_data = ...
nd = NoveltyDetection(model=1)
res = nd.compute_fast_novelty_scores(test_data)
# ...
```
for matlab methods, data should be saved in 'mat' format first:
```python
# normal_data = ..., test_data = ...
nd = NoveltyDetection(model=1)
latent_normal_data = nd.encode(normal_data)
latent_test_data = nd.encode(test_data)
nd.save_to_mat(latent_normal_data, "normal_data_path")
nd.save_to_mat(latent_test_data, "test_data_path")
```
then you can run a matlab code:
```matlab
test_data = load('test_data_path', 'data');
normal_data = load('normal_data_path', 'data');
novelty_score = compute_novelty_score(normal_data, test_data, 'metric', 'euclidean', 'use_gpu', true);
...
```
