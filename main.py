from utils import *
import matplotlib.pyplot as plt
import itertools

########################################################
# Data cases:
#
# DISCRETE:
# discrete_iid_correlated - correlated iid data
# ising_oblivious - ising channel with oblivious encoder, X~Ber(0.5)
# ising_opt - ising with optimal feedback scheme
#
# CONTINUOUS:
# gauss_iid - iid Gaussian data, derived from the ARMA model
# gauss_arma - ARMA Gaussian process
# gauss_delayed - ARMA gauss with increasing weights
########################################################

n = 100000          # num samples
m = 15              # infomat side size
case = 'gauss_delayed'          # data case

Nx = 2              # x alphabet size (discrete)
Ny = 2              # y alphabet size (discrete)
d = 1               # dimension (cont.)

verbose = True
plot = True         # infomat plot value

# generate data:
if case == 'discrete_iid_correlated':
    x,y = gen_data_iid_correlated(n,m,Nx)
elif case == 'ising_oblivious':
    x,y = gen_data_ising_binary_iid(n,m)
elif case == 'ising_opt':
    x,y = gen_data_ising_binary_opt(n, m)
elif case == 'gauss_iid':
    x,y = gen_data_gauss(n,m,'iid')
    # x,y = gen_data_gauss(n,m,case='iid')
elif case == 'gauss_arma':
    x, y = gen_data_gauss(n, m, 'arma')
    # x, y = gen_data_gauss(n, m, case = 'arma')
elif case == 'gauss_delayed':
    x,y = gen_data_gauss(n,m,case='delayed')
else:
    raise ValueError("data case not valid")


if case in ('discrete_iid_correlated','ising_oblivious','ising_opt'):
    est = 'plugin'
else:
    est = 'cov'


infomat = np.zeros(shape=(m,m))
for i,j in itertools.product(range(m), repeat=2):
    if est == 'plugin':
        infomat[i,j] = est_cmi_plugin(x, y, i, j)
    else:
        infomat[i, j] = est_cmi_gauss(x, y, i, j)
    if verbose:
        print(f'finished ({i},{j})')


if plot:
    fig, ax = plt.subplots()
    plt.imshow(infomat, cmap = 'cividis', vmin = 0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xlabel('X', fontsize = 16)
    plt.ylabel('Y', fontsize = 16)
    ax.tick_params(axis = 'x', labelsize = 'large')  # Increase x-axis tick label size
    ax.tick_params(axis = 'y', labelsize = 'large')  # Increase y-axis tick label size
    plt.title(case)
    plt.colorbar()
    plt.show()


