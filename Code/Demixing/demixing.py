import numpy as np
import theano
import theano.tensor as T
from scipy.stats import poisson
from functools import partial
import multiprocessing as mp

nneuron = 61
min_angle = -90
max_angle = 90
sprefs = np.linspace(min_angle, max_angle, nneuron)
eps = np.finfo(np.float64).eps
sigtc_sq = float(10**2)

def cartesian(arrays, out=None, lists=False):
    """Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    if lists:
        out_list = []
        for i in range(len(out)):
            out_list.append([[out[i][0]], [out[i][1]]])
        out = out_list

    return out

def random_s(ndata, sort):
    s = np.random.rand(2, ndata) * 120 - 60
    if sort:
        s = np.sort(s, axis=0)
    return s[0], s[1]

def random_c(ndata, ndims, low, high, sort):
    c_range = high - low
    if ndims == 1:
        c = np.random.rand(ndims, ndata)[0] * c_range + low
    else:
        c = np.random.rand(ndims, ndata) * c_range + low
    if sort:
        c = np.sort(c, axis=0)
    return c
    
def generate_popcode_data(ndata, nneuron, sigtc_sq, r_max, noise, sort, s_0, s_1, c_0, c_1, c_50=13.1):
    c_rms = np.sqrt(np.square(c_0) + np.square(c_1))
    sprefs_data = np.tile(sprefs, (ndata, 1))
    s_0t = np.exp(-np.square((np.transpose(np.tile(s_0, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_0 = c_0 * s_0t.T
    s_1t = np.exp(-np.square((np.transpose(np.tile(s_1, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_1 = c_1 * s_1t.T
    #r = r_max * (stim_0 + stim_1)/(c_50 + c_rms)
    r = r_max * (stim_0 + stim_1)
    r = r.T
    s = np.array((s_0, s_1)).T
    s = s/90
    c = np.array((c_0, c_1)).T
    if noise == "poisson":
        r = np.random.poisson(r) + 0.0
    return r, s, c

def generate_trainset(ndata, highlow=False, discrete_c=None, low=.3, high=.7, r_max=10):
    s_0, s_1 = random_s(ndata, True)
    if highlow:
        if type(discrete_c) == list:
            low = min(discrete_c[0])
            high = max(discrete_c[0])
        c_arr = np.concatenate((np.ones((ndata/2, 2)) * low, np.ones((ndata/2, 2)) * high), axis=0)
        np.random.shuffle(c_arr)
        c_0, c_1 = c_arr.T
    elif discrete_c:
        if type(discrete_c) == int:
            cs = np.linspace(low, high, discrete_c)
            perm_cs = cartesian((cs, cs))
        else:
            perm_cs = cartesian(discrete_c)
        c_arr = np.repeat(perm_cs, ndata/len(perm_cs), axis=0)
        np.random.shuffle(c_arr)
        c_0, c_1 = c_arr.T
        print ndata/len(perm_cs), "trials per contrast level"
        if ndata%len(perm_cs) != 0:
            print "Not divisible, only generated", ndata / len(perm_cs) * len(perm_cs), "trials"
        ndata = ndata / len(perm_cs) * len(perm_cs)
    else:
        c_0, c_1 = np.ones((2, ndata)) * .5
    r, s, c = generate_popcode_data(ndata, nneuron, sigtc_sq, r_max, "poisson", True, s_0, s_1, c_0, c_1)
    return r, s, c

def generate_testset(ndata, stim_0=None, stim_1=None, con_0=None, con_1=None, discrete_c=None, low=.5, high=.5, r_max=10):
    if con_0 is not None:
        c_0 = np.ones(ndata) * con_0
        c_1 = np.ones(ndata) * con_1
    else:
        c_range = high - low
        if discrete_c:
            if type(discrete_c) == int:
                cs = np.linspace(low, high, discrete_c)
                perm_cs = cartesian((cs, cs))
            else:
                perm_cs = cartesian(discrete_c)
            c_0, c_1 = np.repeat(perm_cs, ndata/len(perm_cs), axis=0).T
            print ndata/len(perm_cs), "trials per contrast level"
            if ndata%len(perm_cs) != 0:
                print "Not divisible, only generated", ndata / len(perm_cs) * len(perm_cs), "trials"
            ndata = ndata / len(perm_cs) * len(perm_cs)
        else:
            c_0, c_1 = np.random.rand(2, ndata) * c_range + low
    if not stim_0:
        s_0, s_1 = random_s(ndata, True)
    else:
        s_0, s_1 = np.ones((2, ndata))
        s_0 = s_0 * stim_0
        s_1 = s_1 * stim_1
    r, s, c = generate_popcode_data(ndata, nneuron, sigtc_sq, r_max, "poisson", True, s_0, s_1, c_0, c_1)
    return r, s, c

def generate_trainset_cat(ndata, c_0=4, c_1=1, r_max=1):
    numvec = np.random.binomial(1, .5, size=ndata).astype(int)
    s_0, s_1 = np.random.rand(2, ndata) * 120 - 60
    r, numvec, s, c  = generate_popcode_data_cat(ndata, numvec, nneuron, sigtc_sq, r_max, "poisson", s_0, s_1, c_0, c_1)
    y = s.T[0]
    return r, y, s, c, numvec 

def generate_testset_cat(ndata, stim_0, stim_1, c_0=4, c_1=1, r_max=1):
    numvec = np.random.binomial(1, .5, size=ndata).astype(int)
    s_0 = np.ones(ndata) * stim_0
    s_1 = np.ones(ndata) * stim_1
    r, numvec, s, c  = generate_popcode_data_cat(ndata, numvec, nneuron, sigtc_sq, r_max, "poisson", s_0, s_1, c_0, c_1)
    y = s.T[0]
    return r, y, s, c, numvec 
    
def generate_popcode_data_cat(ndata, numvec, nneuron, sigtc_sq, r_max, noise, s_0, s_1, c_0, c_1):
    c0vec = c_0 * np.ones(ndata)
    c1vec = c_1 * numvec
    c_rms = np.sqrt(np.square(c0vec) + np.square(c1vec))
    sprefs_data = np.tile(sprefs, (ndata, 1))
    s_0t = np.exp(-np.square((np.transpose(np.tile(s_0, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_0 = c0vec * s_0t.T
    s_1t = np.exp(-np.square((np.transpose(np.tile(s_1, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_1 = c1vec * s_1t.T
    r = r_max * (stim_0 + stim_1)
    r = r.T
    s = np.array((s_0, s_1)).T
    s = s/90
    c = np.array((c_0, c_1)).T
    if noise == "poisson":
        r = np.random.poisson(r) + 0.0
    return r, numvec, s, c

def lik_means(s_1, s_2, c_0=.5, c_1=.5, sprefs=sprefs, sigtc_sq=sigtc_sq, r_max=10):
    sprefs_data = np.tile(sprefs, (len(s_1), 1))
    s_0t = np.exp(-np.square((np.transpose(np.tile(s_1, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_0 = c_0 * s_0t.T
    s_1t = np.exp(-np.square((np.transpose(np.tile(s_2, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_1 = c_1 * s_1t.T
    r = r_max * (stim_0 + stim_1)
    return r.T

def posterior(r, means, s1_grid, s2_grid, ret_grid=False):
    ns_liks = poisson.pmf(r, mu=means)
    stim_liks = np.prod(ns_liks, axis=1)
    #p_s = 2/14400
    #logp_s = np.log(p_s)
    logp_s = -3.8573325
    loglik = np.sum(np.log(ns_liks), axis=1)
    grid = np.exp(loglik + logp_s)/np.sum(np.exp(loglik + logp_s))
    mean1 = np.sum(s1_grid * grid)
    mean2 = np.sum(s2_grid * grid)
    expsquare1 = np.sum(np.square(s1_grid) * grid)
    expsquare2 = np.sum(np.square(s2_grid) * grid)
    var1 = expsquare1 - np.square(mean1)
    var2 = expsquare2 - np.square(mean2)
    if ret_grid:
        return mean1, mean2, var1, var2, (s1_grid, s2_grid, grid)
    return mean1, mean2, var1, var2

def posterior_setup(low=.3, high=.7, discrete_c = 3, num_s=100, r_max=10):
    grid = np.linspace(-60, 60, num_s)
    s1s = np.concatenate([[grid[i]]*(num_s-i) for i in range(num_s)])
    s2s = np.concatenate([grid[i:num_s+1] for i in range(num_s)])
    if type(discrete_c) == int:
        cs = np.linspace(low, high, discrete_c)
        s1_grid, c1_grid, c2_grid = cartesian((s1s, cs, cs)).T
        s2_grid = np.repeat(s2s, (discrete_c**2), axis=0)
    else:
        c1 = discrete_c[0]
        c2 = discrete_c[1]
        s1_grid, c1_grid, c2_grid = cartesian((s1s, c1, c2)).T
        s2_grid = np.repeat(s2s, (len(c1) * len(c2)), axis=0)
    means = lik_means(s1_grid, s2_grid, c_0=c1_grid, c_1=c2_grid, r_max=r_max)
    partial_post = partial(posterior, means=means, s1_grid=s1_grid, s2_grid=s2_grid)
    return partial_post

def get_posteriors(r, post_func):
    posteriors = {'mean_s1': None, 'mean_s2': None, 'var_s1': None, 'var_s2': None}
    p = np.array([post_func(r[i]) for i in range(len(r))]).T
    posteriors['mean_s1'], posteriors['mean_s2'], posteriors['var_s1'], posteriors['var_s2'] = p
    return posteriors

def get_posteriors_pool(r, post_func):
    pool = mp.Pool(processes=8)
    posteriors = {'mean_s1': None, 'mean_s2': None, 'var_s1': None, 'var_s2': None}
    p = np.array(pool.map(post_func, r)).T
    posteriors['mean_s1'], posteriors['mean_s2'], posteriors['var_s1'], posteriors['var_s2'] = p
    return posteriors

def posterior_cat(r, means, s1_grid):
    liks = poisson.pmf(r, mu=means)
    #p_s = 2/14400
    #logp_s = np.log(p_s)
    logp_s = -3.8573325
    #p_cat = 1/2
    #logp_cat = np.log(p_cat)
    logp_cat = -0.301029996
    loglik = np.sum(np.log(liks), axis=1)
    mean = np.sum(s1_grid * np.exp(loglik + logp_s + logp_cat)/np.sum(np.exp(loglik + logp_s + logp_cat)))
    expsquare = np.sum(np.square(s1_grid) * np.exp(loglik + logp_s + logp_cat)/np.sum(np.exp(loglik + logp_s + logp_cat)))
    var = expsquare - np.square(mean)
    return mean, var

def get_posteriors_pool_cat(r, post_func):
    pool = mp.Pool(processes=8)
    posteriors = {'mean': None, 'var': None}
    p = np.array(pool.map(post_func, r)).T
    posteriors['mean'], posteriors['var'] = p
    return posteriors

def posterior_setup_cat(high=4, low=1, num_s=100, r_max=10):
    grid = np.linspace(-60, 60, num_s)
    cats = [0, 1]
    s1_grid, s2_grid, cat_grid = cartesian((grid, grid, cats)).T
    means = lik_means_cat(s1_grid, s2_grid, cat_grid, c_0=high, c_1=low, r_max=r_max)
    partial_post = partial(posterior_cat, means=means, s1_grid=s1_grid)
    return partial_post

def lik_means_cat(s_1, s_2, cat, c_0=4, c_1=1, sprefs=sprefs, sigtc_sq=sigtc_sq, r_max=1):
    c0vec = c_0 * np.ones(len(cat))
    c1vec = c_1 * cat
    sprefs_data = np.tile(sprefs, (len(s_1), 1))
    s_0t = np.exp(-np.square((np.transpose(np.tile(s_1, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_0 = c0vec * s_0t.T
    s_1t = np.exp(-np.square((np.transpose(np.tile(s_2, (nneuron, 1))) - sprefs_data))/(2 * sigtc_sq))
    stim_1 = c1vec * s_1t.T
    r = r_max * (stim_0 + stim_1)
    return r.T

def get_statistics(s1, s2, preds):
    mean_s1 = np.mean(preds[0])
    mean_s2 = np.mean(preds[1])
    bias_s1 = mean_s1 - s1
    bias_s2 = mean_s2 - s2
    mse = np.mean((preds[0] - s1)**2 + (preds[1] - s2)**2)
    covmat = np.cov(preds)
    var_s1 = covmat[0, 0]
    var_s2 = covmat[1, 1]
    cov = covmat[0, 1]
    corr = cov / (np.sqrt(var_s1) * np.sqrt(var_s2))
    stats = {'mean_s1': mean_s1, 'mean_s2': mean_s2, 'bias_s1': bias_s1, 'bias_s2': bias_s2, 'var_s1': var_s1, 'var_s2': var_s2, 'cov': cov, 'corr': corr, 'mse': mse}
    return stats

def get_statistics_cat(s, preds):
    mean = np.mean(preds)
    bias = mean - s
    mse = np.mean((preds - s)**2)
    var = np.var(preds)
    stats = {'mean': mean, 'bias': bias, 'var': var, 'mse': mse}
    return stats

"""
Multilayer ReLU net
"""

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, COM=False,
                 activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        if W is None:
            W_values = (1/np.sqrt(n_in)) * np.random.randn(n_in, n_out)
            
            W = theano.shared(value=W_values, name='W', borrow=True)

        if COM:
            self.W = W
            
            ones = np.ones((n_in, n_out))
            
            self.output = T.dot(input, self.W)/T.dot(input, ones)
            
            # parameters of the model
            self.params = [self.W]
        else:
            if b is None:
                b_values = np.zeros((n_out,), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b', borrow=True)

            self.W = W
            self.b = b

            lin_output = T.dot(input, self.W) + self.b
            self.output = (
                lin_output if activation is None
                else activation(lin_output)
            )
            # parameters of the model
            self.params = [self.W, self.b]

class MLP(object):


    def __init__(self, rng, input, n_in, n_hidden, n_out, COM=False):
        """Initialize the parameters for the multilayer perceptron

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=relu
        )
        
        if COM:
            self.hiddenLayer2 = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer1.output,
                n_in=n_hidden,
                n_out=n_out,
                COM=True,
                activation=None
            )
        else:
            self.hiddenLayer2 = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer1.output,
                n_in=n_hidden,
                n_out=n_out,
                activation=None
            )
        
        self.y_pred = self.hiddenLayer2.output
        
        # the parameters of the model are the parameters of the two layers it is made out of
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params
    
    def get_params(self):

        params = {}
        for param in self.params:
            name = param.name
            if name in params:
                name = name, 2
            params[name] = param.get_value()
        return params
    
    def mse(self, y):
        # error between output and target
        if y.ndim == 1:
            se = (self.y_pred.T - y)**2
        else:
            se = T.sum((self.y_pred - y)**2, axis=1)
        return T.mean(se)
        
    
    def valid_mse(self, y):
        if y.ndim == 1:
            se = (self.y_pred.T * 90 - y * 90)**2
        else:
            se = T.sum((self.y_pred * 90 - y * 90)**2, axis=1)
        return T.mean(se)

    
class Perceptron(object):


    def __init__(self, rng, input, n_in, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.layer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_out,
            #activation=T.nnet.sigmoid
            activation=relu
        )
        
        self.y_pred = self.layer.output
        
        # the parameters of the model are the parameters of the two layers it is made out of
        self.params = self.layer.params
        
    def get_params(self):

        params = {}
        for param in self.params:
            name = param.name
            if name in params:
                name = name, 2
            params[name] = param.get_value()
        return params
    
    def mse(self, y):
        # error between output and target
        if y.ndim == 1:
            se = (self.y_pred.T - y)**2
        else:
            se = T.sum((self.y_pred - y)**2, axis=1)
        return T.mean(se)
    
    def valid_mse(self, y):
        return self.mse(y)
        

def shared_dataset(data_xy, borrow=True, no_c=False):
        """ Function that loads the dataset into shared variables
        """
        data_x, data_y = data_xy[:2]
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype='float32'),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype='float32'),
                                 borrow=borrow)
        return shared_x, shared_y

def train_nn(train_dataset, valid_dataset=None, n_hidden=20, learning_rate=0.01, n_epochs=10, batch_size=20, COM=False, linear=False, print_valid=False, mult_ys=True, rho=0, nesterov=True, mu=0, n_in=61, n_out=2):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

   """
    train_set_x, train_set_y = shared_dataset(train_dataset)
    if valid_dataset:
        valid_set_x, valid_set_y = shared_dataset(valid_dataset)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    #print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.fmatrix('x')   # input data from visual neurons
    if n_out == 1:
        y = T.fvector('y') # ground truth
    else:
        y = T.fmatrix('y')  # ground truth

    rng = np.random.RandomState(1234)

    # construct the MLP class
    if linear:
        nn = Perceptron(rng=rng, input=x, n_in=n_in, n_out=n_out)
    elif COM:
        nn = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out, COM=True)
    else:
        nn = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    
    cost = nn.mse(y)
    
    def RMSprop(cost, params, learning_rate=0.001, rho=0.9, epsilon=1e-6, mu=0, nesterov=False):
        gparams = T.grad(cost, params)
        updates = []
        for p, g in zip(params, gparams):
            v = theano.shared(p.get_value() * 0.)
            ms = theano.shared(p.get_value() * 0.)
            ms_new = rho * ms + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(ms_new + epsilon)
            g = g / gradient_scaling
            """
            (1) v_t = mu * v_t-1 - lr * gradient_f(params_t)
            or
            classic
            (2) params_t = params_t-1 + v_t
            nesterov
            (7) params_t = params_t-1 + mu * v_t - lr * gradient_f(params_t-1)
            (8) params_t = params_t-1 + mu**2 * v_t-1 - (1+mu) * lr * gradient_f(params_t-1)
            """
            v_new = mu * v - (1 - mu) * learning_rate * g
            if nesterov:
                p_new = p + mu * v_new - (1 - mu) * learning_rate * g
            else:
                p_new = p + v_new
            updates.append((ms, ms_new))
            updates.append((v, v_new))
            updates.append((p, p_new))
                
        return updates
    
    if rho:
        updates = RMSprop(cost, nn.params, learning_rate=learning_rate, rho=rho, mu=mu, nesterov=nesterov)
    else:
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in nn.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(nn.params, gparams)
        ]
    
    def inspect_inputs(i, node, fn):
        print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]

    def inspect_outputs(i, node, fn):
        print "output(s) value(s):", [output[0] for output in fn.outputs]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    if mult_ys:
        valid_mse = nn.valid_mse(y)
    else:
        valid_mse = cost
    
    validate_model = theano.function(
        inputs=[index],
        outputs=valid_mse,
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    #print '... training'

    epoch = 0 

    if valid_dataset:
        valid_mse = np.zeros(n_epochs)

    while (epoch < n_epochs):
        for minibatch_index in xrange(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index)
            
        if valid_dataset:
            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            valid_mse[epoch] = this_validation_loss
            if print_valid:
                print(
                    'epoch %i, validation error %f' %
                    (
                        epoch,
                        this_validation_loss,
                    )
                )
            
        epoch = epoch + 1

    if valid_dataset:
        return nn, x, valid_mse
    return nn, x

def test_nn(nn, nnx, test_data):
    #print 'testing'
    test_batch_size = 1
    test_set_x, test_set_y = shared_dataset(test_data)
    index = T.lscalar()  # index to a [mini]batch
    x = nnx   # input data from visual neurons
    test_model = theano.function(
        inputs=[index],
        outputs=nn.y_pred,
        givens={
            x: test_set_x[index * test_batch_size: (index + 1) * test_batch_size]
        },
    )
    
    true_ys = test_set_y.get_value()
    pred_ys = np.zeros(true_ys.shape)
    for i in range(len(true_ys)):
        pred_ys[i] = test_model(i)
        #print test_model(i)[0], true_ys[i]
        #print test_model(i)[0] * 90, true_ys[i]
    
    #print nn.get_params()
    return pred_ys, true_ys

def python_relu(x):
    return x * (x > 0)

def get_hu_responses(r, nn):
    nn_params = nn.get_params()
    b = nn_params['b']
    W = nn_params['W']
    trials = python_relu(np.dot(r, W) + b)
    return trials

def get_mean_acts(s_arr, nn, c, hus=20):
    l_sarr = len(s_arr)
    acts = np.zeros((l_sarr, l_sarr, hus))
    perm_cs = cartesian(c)
    for i in range(l_sarr):
        for j in range(i+1, l_sarr):
            s1 = s_arr[i]
            s2 = s_arr[j]
            r, _, _ = generate_testset(len(perm_cs) * 50, stim_0=s1, stim_1=s2, discrete_c=c, r_max=1)
            acts[i][j] = np.mean(get_hu_responses(r, nn), axis=0)
    return np.array(acts)

def get_corrs(nn, posts_v1, posts_v2, rand_nn=False):
    lin_corrs_ins = np.zeros((9, 200))
    lin_corrs_hus = np.zeros((9, 200))
    lin_corrs_ins_opt = np.zeros((9, 200))
    lin_corrs_hus_opt = np.zeros((9, 200))
    kurt_corrs = np.zeros((9, 200))
    sum_corrs = np.zeros((9, 200))

    lin_corrs_ins_all = np.zeros(200)
    lin_corrs_hus_all = np.zeros(200)
    kurt_corrs_all = np.zeros(200)
    sum_corrs_all = np.zeros(200)

    keys = posts_v1.keys()     
    for i in range(200):
        print i
        validset_ins = {}
        trainset_ins = {}
        train_ins_0 = []
        train_ins_1 = []
        valid_ins_0 = []
        valid_ins_1 = []
        validset_hus = {}
        trainset_hus = {}
        train_hus_0 = []
        train_hus_1 = []
        valid_hus_0 = []
        valid_hus_1 = []
        for k in range(len(keys)):
            k_i = keys[k]
            x = testsets[k_i]
            y = np.array((1/posts_v1[k_i], 1/posts_v2[k_i])).T
            inds = range(len(x))
            np.random.shuffle(inds)
            x_shuf = x[inds]
            y_shuf = y[inds]
            if rand_nn:
                x_hus = python_relu(np.dot(x_shuf, nn['W']) + nn['b'])
            else:
                x_hus = get_hu_responses(x_shuf, nn)  
            validset_ins[k] = x_shuf[0:2000], y_shuf[0:2000]
            validset_hus[k] = x_hus[0:2000], y_shuf[0:2000]
            trainset_ins[k] = x_shuf[2000:], y_shuf[2000:]
            trainset_hus[k] = x_hus[2000:], y_shuf[2000:]
            valid_ins_0.append(x_shuf[0:2000])
            valid_ins_1.append(y_shuf[0:2000])
            valid_hus_0.append(x_hus[0:2000])
            valid_hus_1.append(y_shuf[0:2000])
            train_ins_0.append(x_shuf[2000:])
            train_ins_1.append(y_shuf[2000:])
            train_hus_0.append(x_hus[2000:])
            train_hus_1.append(y_shuf[2000:])
        trainset_ins_all = np.concatenate(train_ins_0), np.concatenate(train_ins_1)
        trainset_hus_all = np.concatenate(train_hus_0), np.concatenate(train_hus_1)
        validset_ins_all = np.concatenate(valid_ins_0), np.concatenate(valid_ins_1)
        validset_hus_all = np.concatenate(valid_hus_0), np.concatenate(valid_hus_1) 

        weights_ins = np.linalg.lstsq(trainset_ins_all[0], trainset_ins_all[1])[0]
        weights_hus = np.linalg.lstsq(trainset_hus_all[0], trainset_hus_all[1])[0]
        for v in range(len(validset_hus)):
            weights_ins_opt = np.linalg.lstsq(trainset_ins[v][0], trainset_ins[v][1])[0]
            weights_hus_opt = np.linalg.lstsq(trainset_hus[v][0], trainset_hus[v][1])[0]
            
            hus, vpost = validset_hus[v]
            inputs, vpost = validset_ins[v]
            lin_preds_hus = np.dot(hus, weights_hus)
            lin_preds_ins = np.dot(inputs, weights_ins)
            lin_preds_hus_opt = np.dot(hus, weights_hus_opt)
            lin_preds_ins_opt = np.dot(inputs, weights_ins_opt)
            kurt_preds = kurtosis(hus, axis=1)
            sum_preds = np.sum(hus, axis=1)
            vp = np.concatenate((vpost.T[0], vpost.T[1]))
            lin_preds_ins = np.concatenate((lin_preds_ins.T[0], lin_preds_ins.T[1]))
            lin_preds_hus = np.concatenate((lin_preds_hus.T[0], lin_preds_hus.T[1]))
            lin_preds_ins_opt = np.concatenate((lin_preds_ins_opt.T[0], lin_preds_ins_opt.T[1]))
            lin_preds_hus_opt = np.concatenate((lin_preds_hus_opt.T[0], lin_preds_hus_opt.T[1]))
            kurt_preds = np.concatenate((kurt_preds, kurt_preds))
            sum_preds = np.concatenate((sum_preds, sum_preds))

            lin_corrs_ins[v][i] = np.corrcoef(vp, lin_preds_ins)[0, 1]
            lin_corrs_hus[v][i] = np.corrcoef(vp, lin_preds_hus)[0, 1]
            lin_corrs_ins_opt[v][i] = np.corrcoef(vp, lin_preds_ins_opt)[0, 1]
            lin_corrs_hus_opt[v][i] = np.corrcoef(vp, lin_preds_hus_opt)[0, 1]
            kurt_corrs[v][i] = np.corrcoef(vp, kurt_preds)[0, 1]
            sum_corrs[v][i] = np.corrcoef(vp, sum_preds)[0, 1]
        
        hus, vpost = validset_hus_all
        inputs, vpost = validset_ins_all
        lin_preds_ins = np.dot(inputs, weights_ins)
        lin_preds_hus = np.dot(hus, weights_hus)
        kurt_preds = kurtosis(hus, axis=1)
        sum_preds = np.sum(hus, axis=1)
        vp = np.concatenate((vpost.T[0], vpost.T[1]))
        lin_preds_ins = np.concatenate((lin_preds_ins.T[0], lin_preds_ins.T[1]))
        lin_preds_hus = np.concatenate((lin_preds_hus.T[0], lin_preds_hus.T[1]))
        kurt_preds = np.concatenate((kurt_preds, kurt_preds))
        sum_preds = np.concatenate((sum_preds, sum_preds))

        lin_corrs_ins_all[i] = np.corrcoef(vp, lin_preds_ins)[0, 1]
        lin_corrs_hus_all[i] = np.corrcoef(vp, lin_preds_hus)[0, 1]
        kurt_corrs_all[i] = np.corrcoef(vp, kurt_preds)[0, 1]
        sum_corrs_all[i] = np.corrcoef(vp, sum_preds)[0, 1]

        return lin_corrs_ins, lin_corrs_hus, lin_corrs_ins_opt, lin_corrs_hus_opt, kurt_corrs, sum_corrs, lin_corrs_ins_all, lin_corrs_hus_all, kurt_corrs_all, sum_corrs_all

