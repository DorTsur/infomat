import numpy as np
import itertools

# DATA FNS
def gen_data_iid_correlated(n, m, Nx):
    x = np.random.randint(0, Nx, (n, m))
    y = (x + np.random.binomial(1, 0.1, (n, m))) % Nx
    return x,y

def gen_data_ising_binary_iid(n,m):
    s = np.zeros((n, 1), dtype = int)  # Starting with s as all zeros
    x_sequence = np.random.binomial(1, 0.5, (n, m))  # Sequence of x elements
    y_sequence = np.zeros((n, m), dtype = int)  # To store the sequence of y elements

    for i in range(m):
        x = x_sequence[:, i:i + 1]  # Current batch of x

        # y is x with probability 0.5, or s with probability 0.5
        y = np.where(np.random.rand(n, 1) < 0.5, x, s)

        # Update y_sequence
        y_sequence[:, i:i + 1] = y

        # Update s to current x and s
        s = x


    return x_sequence, y_sequence

def gen_data_ising_binary_opt(n, m):
    # Define input shape and initialize parameters
    input_shape = (n, 1)
    logits = np.concatenate([0.4503 * np.ones(input_shape), (1 - 0.4503) * np.ones(input_shape)], axis = 1)
    s = np.zeros(input_shape, dtype = int)
    q = np.ones(input_shape, dtype = int)

    x_l = []
    y_l = []

    for _ in range(m):
        batch_size = input_shape[0]
        # Select logits based on current state
        cur_logits = logits[np.arange(batch_size), s[:, 0].astype(int)]
        cur_logits = np.stack([cur_logits, 1 - cur_logits], axis = 1)
        # Generate new symbols based on current logits
        new_symbol = np.array([np.random.choice([0, 1], p = logit) for logit in cur_logits])
        x = np.where(q == 0, s, new_symbol[:, None])
        # Introduce channel noise
        channel_noise = np.random.randint(0, 2, size = input_shape)
        y = np.where(channel_noise == 1, x, s)
        s_plus = x
        q_plus = np.where(q == 1, np.where(s == y, 0, 1), 1)
        s = s_plus
        q = q_plus
        x_l.append(x)
        y_l.append(y)

    # Concatenate along time dimension and then remove the last dimension
    x_l = np.concatenate(x_l, axis = 1)
    y_l = np.concatenate(y_l, axis = 1)

    return x_l, y_l

def gen_data_gauss(n, m,case='iid'):
    X = np.zeros([n, m])
    Y = np.zeros([n, m])

    if case == 'iid':
        # Generate data using the iid method
        for i in range(m):  # Loop over m instead of n
            temp_X = np.random.normal(scale = 1, size = (n,))  # Generate [n,] shape directly
            temp_Y = temp_X + np.random.normal(scale = 1, size = (n,))  # Generate [n,] shape directly
            X[:, i] = temp_X  # Fill column i
            Y[:, i] = temp_Y  # Fill column i
        return X, Y

    if case == 'delayed':
        # Generate data using the ARMA method
        alpha_x = alpha_y = np.linspace(0, 0.001, m)
        beta_x = beta_y = np.linspace(0, 0.3, m)
    else:
        alpha_x = alpha_y = beta_x = beta_y = [0.5]*m

    for i in range(m):
        X[:, i] += np.random.normal(scale = 0.1, size = (n,))
        Y[:, i] += np.random.normal(scale = 0.1, size = (n,))
        for j in range(i+1):  # Fix to loop over previous values
            # Update each element in column i based on past values
            X[:, i] += alpha_x[j] * X[:, i - j] + beta_x[j] * Y[:, i - j]
            Y[:, i] += alpha_y[j] * X[:, i - j] + beta_y[j] * Y[:, i - j]

    return X, Y


# ESTIMATOR FNS:

def est_cmi_plugin(X,Y,i,j):
    Infomat = 0

    x, y = None, None
    # H(X^{i-1},Y^{j})
    i += 1
    j += 1
    if i > 1:
        x = X[:, :i-1]
    y = Y[:, :j]
    h = est_ent_plugin(x, y)
    Infomat += h

    x, y = None, None
    # H(X^{i-1},Y^{j-1})
    if i > 1:
        x = X[:, :i - 1]
    if j > 1:
        y = Y[:, :j - 1]
    h = est_ent_plugin(x, y)
    Infomat -= h

    x, y = None, None
    # H(X^{i},Y^{j})
    x = X[:, :i]
    y = Y[:, :j]
    h = est_ent_plugin(x, y)
    Infomat -= h

    x, y = None, None
    # H(X^{i},Y^{j-1})
    x = X[:, :i]
    if j > 1:
        y = Y[:, :j - 1]
    h = est_ent_plugin(x, y)
    Infomat += h

    return Infomat

def est_ent_plugin(x,y):
    sample = []
    if x is not None:
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis = -1)
        sample.append(x)
    if y is not None:
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis = -1)
        sample.append(y)
    if sample == []:
        return 0

    sample = np.concatenate(sample, axis=-1)
    m = sample.shape[0]  # Number of rows

    # decimals = np.fromiter((int("".join(map(str, row)), 2) for row in sample), dtype = np.int32, count = m)
    decimals = np.fromiter((int("".join(map(str, row)), 2) for row in sample), dtype = np.int32)
    unique, counts = np.unique(decimals, return_counts = True)
    probabilities = counts.astype('float32') / len(sample)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy



def est_cmi_gauss(X,Y,i,j):
    Infomat = 0

    x, y = None, None
    # H(X^{i-1},Y^{j})
    i += 1
    j += 1
    if i > 1:
        x = X[:, :i-1]
    y = Y[:, :j]
    h = est_ent_gauss(x, y)
    Infomat += h

    x, y = None, None
    # H(X^{i-1},Y^{j-1})
    if i > 1:
        x = X[:, :i - 1]
    if j > 1:
        y = Y[:, :j - 1]
    h = est_ent_gauss(x, y)
    Infomat -= h

    x, y = None, None
    # H(X^{i},Y^{j})
    x = X[:, :i]
    y = Y[:, :j]
    h = est_ent_gauss(x, y)
    Infomat -= h

    x, y = None, None
    # H(X^{i},Y^{j-1})
    x = X[:, :i]
    if j > 1:
        y = Y[:, :j - 1]
    h = est_ent_gauss(x, y)
    Infomat += h

    return Infomat

def est_ent_gauss(x,y):
    sample = []
    if x is not None:
        sample.append(x)
    if y is not None:
        sample.append(y)
    if sample == []:
        return 0
    sample = np.concatenate(sample,axis=1)
    N = sample.shape[0]
    n = sample.shape[1]
    Sigma = 1/(N-1)*np.matmul(np.transpose(sample),sample) + 1e-5*np.eye(n)
    mi = 0.5*np.log(np.linalg.det(Sigma))
    if mi == 0:
        return 0
    else:
        return mi


