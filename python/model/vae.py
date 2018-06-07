# ############### Variational Autoencoder ####################
# This is an adapted implementation of vae for novelty detection from
# https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py

import time

import lasagne as nn
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from util import get_model_params, get_training_params, model_path

from python.model import TrainData


# ##################### Custom layer for middle of VAE ######################
# This layer takes the mu and sigma (both DenseLayers) and combines them with
# a random vector epsilon to sample values for a multivariate Gaussian

class GaussianSampleLayer(nn.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1, 2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape = (self.input_shapes[0][0] or inputs[0].shape[0],
                 self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)


# ############################## Build Model #################################
# encoder has #depth hidden layer, where we get mu and sigma for Z given an inp X
# continuous decoder has #depth hidden layer, where we get reconstruction for X given Z

def build_vae(inputvar, n_channels, depth, z_dim, n_hid_first, L=5):
    """
    :param inputvar:
    :param n_channels: number of channels in the input vector
    :param depth: depth of the encoder and decoder of the VAE
    :param z_dim: dimensionality of the latent space
    :param n_hid_first: number of neurons in the first hidden layer of the encoder.
    For each respective layer of the encoder number of layers is twice less than the
    number of layers in the previous layer. Decoder is symmetric to the encoder.
    :param L: number of samples from latent space to compute output values
    :return:
    """
    # encoder
    l = nn.layers.InputLayer(shape=(None, n_channels),
                             input_var=inputvar, name='input')

    # encoder hidden layers
    for i in range(depth):
        num_units = int(n_hid_first / (2 ** i))
        l = nn.layers.DenseLayer(l, num_units=num_units,
                                 nonlinearity=nn.nonlinearities.rectify, name='enc_hid' + str(i))

    l_enc_mean = nn.layers.DenseLayer(l, num_units=z_dim,
                                      nonlinearity=None, name='enc_mu')

    l_enc_logsigma = nn.layers.DenseLayer(l, num_units=z_dim,
                                            nonlinearity=None, name='enc_logsigma')

    # decoder
    l_dec = {}
    l_dec_mean_list = []
    l_dec_logsigma_list = []
    l_x_list = []
    # tie the weights of all L versions so they are the "same" layer
    W_dec_hid = [None] * depth
    b_dec_hid = [None] * depth
    W_dec_mean = None
    b_dec_mean = None
    W_dec_ls = None
    b_dec_ls = None
    for i in range(L):
        l_dec[0] = GaussianSampleLayer(l_enc_mean, l_enc_logsigma, name='Z')
        for j in range(depth):
            num_units = int(n_hid_first / (2 ** (depth - i - 1)))
            l_dec[j+1] = nn.layers.DenseLayer(l_dec[j], num_units=num_units,
                                             nonlinearity=nn.nonlinearities.rectify,
                                             W=nn.init.GlorotUniform() if W_dec_hid[j] is None else W_dec_hid[j],
                                             b=nn.init.Constant(0.) if b_dec_hid[j] is None else b_dec_hid[j],
                                             name='dec_hid' + str(j))

        l_dec_mu = nn.layers.DenseLayer(l_dec[depth], num_units=n_channels,
                                        nonlinearity=None,
                                        W=nn.init.GlorotUniform() if W_dec_mean is None else W_dec_mean,
                                        b=nn.init.Constant(0) if b_dec_mean is None else b_dec_mean,
                                        name='dec_mu')

        # relu_shift is for numerical stability - if training data has any
        # dimensions where stdev=0, allowing logsigma to approach -inf
        # will cause the loss function to become NAN. So we set the limit
        # stdev >= exp(-1 * relu_shift)
        relu_shift = 10

        l_dec_logsigma = nn.layers.DenseLayer(l_dec[depth], num_units=n_channels,
                                              nonlinearity=lambda a: T.nnet.relu(a+relu_shift)-relu_shift,
                                              W=nn.init.GlorotUniform() if W_dec_ls is None else W_dec_ls,
                                              b=nn.init.Constant(0) if b_dec_ls is None else b_dec_ls,
                                              name='dec_logsigma')

        l_x = GaussianSampleLayer(l_dec_mu, l_dec_logsigma,
                                       name='dec_output')
        l_dec_mean_list.append(l_dec_mu)
        l_dec_logsigma_list.append(l_dec_logsigma)
        l_x_list.append(l_x)
        if W_dec_mean is None:
            for j in range(depth):
                W_dec_hid[j] = l_dec[j+1].W
                b_dec_hid[j] = l_dec[j+1].b
            W_dec_mean = l_dec_mu.W
            b_dec_mean = l_dec_mu.b
            W_dec_ls = l_dec_logsigma.W
            b_dec_ls = l_dec_logsigma.b
    l_x = nn.layers.ElemwiseSumLayer(l_x_list, coeffs=1. / L, name='x')

    return l_enc_mean, l_enc_logsigma, l_dec_mean_list, l_dec_logsigma_list, l_x_list, l_x


def log_likelihood(tgt, mu, ls):
    return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + ls)
            - 0.5 * T.sqr(tgt - mu) / T.exp(2 * ls))


def train_network(model):
    n_channels, depth, z_dim, n_hid_first, lam, L = get_model_params(model)
    batch_size, num_epochs, learning_rate = get_training_params(model)

    data = TrainData(batch_size)
    input_var = T.matrix('inputs')

    # Create VAE model
    l_z_mean, l_z_logsigma, l_x_mean_list, l_x_logsigma_list, l_x_list, l_x = \
        build_vae(input_var, n_channels=n_channels, depth=depth, z_dim=z_dim,
                                              n_hid_first=n_hid_first, L=L)

    def build_loss(deterministic):
        layer_outputs = nn.layers.get_output([l_z_mean, l_z_logsigma] + l_x_mean_list
                                             + l_x_logsigma_list,
                                             deterministic=deterministic)
        z_mean = layer_outputs[0]
        z_ls = layer_outputs[1]
        x_mean = layer_outputs[2: 2 + L]
        x_logsigma = layer_outputs[2 + L : 2 + 2 * L]

        # Loss function:  - log p(x|z) + KL_div
        kl_div = lam * 0.5 * T.sum(T.exp(2 * z_ls) + T.sqr(z_mean) - 1 - 2 * z_ls)

        logpxz = sum(log_likelihood(input_var.flatten(2), mu, ls)
                     for mu, ls in zip(x_mean, x_logsigma)) / L
        prediction = x_mean[0] if deterministic else T.sum(x_mean, axis=0) / L
        loss = -logpxz + kl_div

        return loss, prediction

    loss, _ = build_loss(deterministic=False)
    test_loss, test_prediction = build_loss(deterministic=True)

    # ADAM updates
    params = nn.layers.get_all_params(l_x, trainable=True)
    updates = nn.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var], loss, updates=updates)
    val_fn = theano.function([input_var], test_loss)

    previous_val_err_1 = float('inf')
    previous_val_err_2 = float('inf')
    for epoch in range(num_epochs):
        train_err = 0.0
        epoch_size = 0
        start_time = time.time()
        for i in range(data.train_size):
            batch = data.next_batch()
            this_err = train_fn(batch)
            train_err += this_err
            epoch_size += batch.shape[0]

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("training loss: {:.6f}".format(train_err / epoch_size))
        val_err = 0.0
        val_size = 0
        test_data = data.validation_data()
        for i in range(data.validation_size):
            err = val_fn(test_data[i])
            val_err += err
            val_size += test_data[i].shape[0]

        print("validation loss: {:.6f}".format(val_err / val_size))

        # early stopping
        if val_err > previous_val_err_1 and val_err > previous_val_err_2:
            break
        else:
            previous_val_err_2 = previous_val_err_1
            previous_val_err_1 = val_err

        # save the parameters so they can be loaded for next time
        np.savez(model_path(model) + str(epoch), *nn.layers.get_all_param_values(l_x))

        # output samples
        samples = data.validation_samples()
        pred_fn = theano.function([input_var], test_prediction)
        X_pred = pred_fn(samples)
        for i in range(len(samples)):
            print(samples[i] - X_pred[i])


if __name__ == '__main__':
    train_network(1)