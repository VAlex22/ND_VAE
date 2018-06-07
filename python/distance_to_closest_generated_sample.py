# Separate script, to use python multiprocessing, that utilizes pickle #
import sys
from multiprocessing import Pool, cpu_count
import lasagne as nn
import numpy as np
import theano
import theano.tensor as T
from scipy import optimize

from python.util import get_model_params, model_path, model_epoch
from python.model.vae import build_vae

test_data_path, model, bound, output_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

n_channels, depth, z_dim, n_hid_first, lam, L = get_model_params(model)
test_data = np.load(test_data_path)

# load trained model
input_var = T.matrix('inputs')
z_var = T.vector()
l_z_mean, l_z_stddev, _, _, _, l_x = build_vae(input_var, n_channels=n_channels, depth=depth, z_dim=z_dim,
                                      n_hid_first=n_hid_first, L=1)

with np.load(model_path(model) + str(model_epoch(model)) + '.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

nn.layers.set_all_param_values(l_x, param_values)

# create encoder function to find initial values for z
encoder = nn.layers.get_output([l_z_mean, l_z_stddev], deterministic=True)
encode = theano.function([input_var], encoder)

# create decoder function
generated_x = nn.layers.get_output(l_x, {l_z_mean: z_var}, deterministic=True)
gen_fn = theano.function([z_var], generated_x)

# create l2 loss to optimize over latent space
z_mean, z_stddev = encode(test_data)
z_0 = z_mean


def loss(z, voxel):
    x = gen_fn(z).reshape(n_channels)
    return np.linalg.norm(voxel-x)

if bound == 0:
    def minimize_voxel(args):
        loss, z_0, voxel = args
        optimize_result = optimize.minimize(loss, z_0, voxel)
        return loss(optimize_result.x, voxel)
else:
    boundaries = ((-bound, bound),)
    for _ in range(z_dim-1):
        boundaries += ((-bound, bound),)

    def minimize_voxel(args):
        loss, z_0, voxel = args
        optimize_result = optimize.minimize(loss, z_0, voxel, bounds=boundaries)
        return loss(optimize_result.x, voxel)

args = [(loss, z_0[i], test_data[i]) for i in range(len(test_data))]
p = Pool(cpu_count())
novelty_score = np.array(p.map(minimize_voxel, args))
np.save(output_path, novelty_score)
