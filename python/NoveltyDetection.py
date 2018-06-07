import lasagne as nn
import numpy as np
import theano
import theano.tensor as T
from scipy.io import savemat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from python.util import get_model_params, get_training_params, model_epoch, model_path
from python.model.vae import build_vae


class NoveltyDetection:
    def __init__(self, model):
        """
        :param model: number of the model in the resources
        """
        self.model = model
        self.input = T.matrix('inputs')
        self.z_m, self.z_ls, self.x_m, self.x_ls, _, _ = self.load_model(1, self.input)

    def load_model(self, L, input_var):
        """
        :param L: number of samples from latent space to reconstruct from
        :return: outputs of the vae
        """
        self.n_channels, depth, self.z_dim, n_hid_first, self.lam, _ = get_model_params(self.model)
        batch_size, _, _ = get_training_params(self.model)

        # load trained model
        l_z_mean, l_z_log_sigma, l_x_mean, l_x_log_sigma, l_x_list, l_x = \
            build_vae(input_var, n_channels=self.n_channels, depth=depth,
                      z_dim=self.z_dim, n_hid_first=n_hid_first, L=L)
        with np.load(model_path(self.model) + str(model_epoch(self.model)) + '.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        nn.layers.set_all_param_values(l_x, param_values)

        return l_z_mean, l_z_log_sigma, l_x_mean, l_x_log_sigma, l_x_list, l_x

    def encode(self, data):
        encoder = nn.layers.get_output([self.z_m, self.z_ls], deterministic=True)
        encode = theano.function([self.input], encoder)
        return encode(data)

    def reconstruct(self, data, deterministic=True):
        reconstructor = nn.layers.get_output(self.x_m + self.x_ls, deterministic=deterministic)
        reconstruct = theano.function([self.input], reconstructor)
        return reconstruct(data)

    def compute_fast_novelty_scores(self, test_data, n_samples=45):
        """
        :param test_data: test data to compute novelty scores on
        :param n_samples: number of samples from normal distribution used for
        stochastic reconstructions
        :return: dictionary containing novelty scores for VAE-regularizer metric and for all
        reconstruction based metrics. Novelty score as a full VAE loss can be computed by
        adding VAE-regularizer novelty score to one of the reconstruction likelihood scores.
        """

        novelty_scores = {}

        # VAE regularizer
        z_m, z_ls = self.encode(test_data)
        novelty_scores["vae-reg"] = kl_div(z_m, z_ls, self.lam)

        # Deterministic reconstruction error
        x_m, x_ls = self.reconstruct(test_data, deterministic=True)
        novelty_scores["dre"] = euclidean_distance(x_m, test_data)

        # Deterministic reconstruction likelihood
        novelty_scores["drl"] = log_lh(x_m, x_ls, test_data)

        # Decoder stochastic reconstruction error
        ns = np.zeros((len(test_data), n_samples))
        for i in range(n_samples):
            x_sampled = sample_from_normal(x_m, x_ls)
            ns[:, i] = euclidean_distance(test_data, x_sampled)

        novelty_scores["dsre"] = np.mean(ns, axis=-1)

        # Stochastic encoder
        ns_det_dec = np.zeros((len(test_data), n_samples))
        ns_ll = np.zeros((len(test_data), n_samples))
        ns_st_dec = np.zeros((len(test_data), n_samples))

        for i in range(n_samples):
            x_m, x_ls = self.reconstruct(test_data, deterministic=False)
            ns_det_dec[:, i] = euclidean_distance(test_data, x_m)
            ns_ll[:, i] = log_lh(x_m, x_ls, test_data)
            x_sampled = sample_from_normal(x_m, x_ls)
            ns_st_dec[:, i] = euclidean_distance(test_data, x_sampled)

        # Encoder stochastic reconstruction error
        novelty_scores["esre"] = np.mean(ns_det_dec, axis=-1)

        # Encoder stochastic reconstruction likelihood
        novelty_scores["esrl"] = np.mean(ns_ll, axis=-1)

        # Fully stochastic reconstruction error
        novelty_scores["fsre"] = np.min(ns_ll, axis=-1)

        return novelty_scores

    def save_to_mat(self, data, path):
        # transform from logsigma to sigma
        data[:, :, 1] = np.exp(data[:, :, 1])
        savemat(path, data=data)


def euclidean_distance(x, y):
    x_ = T.matrix('x')  # candidates
    y_ = T.matrix('y')  # targets
    n_s = T.sqrt(T.sum(T.sqr(x_ - y_), axis=-1))
    dist_fn = theano.function([x_, y_], n_s, profile=True)
    return dist_fn(x, y)


def kl_div(mean, log_sigma, lam):
    ls = T.matrix('ls')
    m = T.matrix('m')
    kl_div = lam * 0.5 * T.sum(T.exp(2 * ls) + T.sqr(m) - 1 - 2 * ls, axis=-1)
    kl_fn = theano.function([ls, m], kl_div, profile=True)
    return kl_fn(log_sigma, mean)


def log_lh(mean, log_sigma, y):
    y_ = T.matrix('y')
    ls = T.matrix('ls')
    m = T.matrix('m')
    llh = T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + ls)
          - 0.5 * T.sqr(y_ - m) / T.exp(2 * ls), axis=-1)
    llh_fn = theano.function([y_, ls, m], llh, profile=True)
    return llh_fn(y, log_sigma, mean)


def sample_from_normal(mean, log_sigma):
    rng = RandomStreams(nn.random.get_rng().randint(1, 2147462579))
    m = T.matrix('mu')
    ls = T.matrix('logsigma')
    samp = m + T.exp(ls) * rng.normal(mean.shape)
    samp_fn = theano.function([m, ls], samp, profile=True)
    return samp_fn(mean, log_sigma)


def main():
    # Example of usage
    # test_data = ...
    nd = NoveltyDetection(1)
    res = nd.compute_fast_novelty_scores(test_data)
    # ...

if __name__ == '__main__':
    main()