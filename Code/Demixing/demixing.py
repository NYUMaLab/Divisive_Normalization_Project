import numpy as np
import math
import os
import sys
import time
import theano
import theano.tensor as T
import pystan
import matplotlib.pyplot as plt
import argparse
import getopt

nneuron = 61
min_angle = -90
max_angle = 90
sprefs = np.linspace(min_angle, max_angle, nneuron)
ndata = 3000

r_max = 10
sigtc_sq = float(10**2)
sigtc = 10
c_50 = 13.1

def random_s(ndata, sort):
    s = np.random.rand(2, ndata) * 120 - 60
    if sort:
        s = np.sort(s, axis=0)
    return s[0], s[1]

def generate_trainset(ndata):
    s_0, s_1 = random_s(ndata, True)
    c_0, c_1 = np.ones((2, ndata)) * .5
    r, s, c = generate_popcode_data(ndata, nneuron, sigtc_sq, c_50, r_max, "poisson", True, s_0, s_1, c_0, c_1)
    return r, s, c

def generate_s1set(ndata):
    s_0, s_1 = random_s(ndata, True)
    c_0 = np.ones(ndata)
    c_1 = np.zeros(ndata)
    r, s, c = generate_popcode_data(ndata, nneuron, sigtc_sq, c_50, r_max, "poisson", True, s_0, s_1, c_0, c_1)
    return r, s, c
    
def generate_popcode_data(ndata, nneuron, sigtc_sq, c_50, r_max, noise, sort, s_0, s_1, c_0, c_1):
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

def generate_s_data(s_0, s_1):
    ndata = 3000
    c_0, c_1 = np.ones((2, ndata)) * .5
    s_0, s_1 = np.ones((2, ndata))
    s_0 = s_0 * -50
    s_1 = s_1 * 15
    r, s, c = generate_popcode_data(ndata, nneuron, sigtc_sq, c_50, r_max, "poisson", True, s_0, s_1, c_0, c_1)
    return r, s, c

def fisher_inf(s_0, s_1, c_0, c_1):
    fs_0 = np.exp(-np.square((np.transpose(np.tile(s_0, (nneuron, 1))) - sprefs))/(2 * sigtc_sq))[0]
    qs_0 = r_max * contrast_0 * fs_0
    df_s0 = ((-s_0 + sprefs)/sigtc_sq) * qs_0
    fs_1 = np.exp(-np.square((np.transpose(np.tile(s_1, (nneuron, 1))) - sprefs))/(2 * sigtc_sq))[0]
    qs_1 = r_max * contrast_1 * fs_1
    df_s1 = ((-s_1 + sprefs)/sigtc_sq) * qs_1
    Q = qs_0 + qs_1
    Q_inv = 1/Q
    J_11 = np.sum(np.square(df_s0) * Q_inv)
    J_22 = np.sum(np.square(df_s1) * Q_inv)
    J_12 = J_21 = np.sum(df_s0 * df_s1 * Q_inv)
    return J_11, J_22, J_12, J_21

def fit_optimal(r, init, sm, N=61, sprefs=sprefs, c_1=.5, c_2=.5, c_50=13.1, r_max=10, c_rms=0.707106781, sig_tc=10, sigtc_sq=10**2):
    neurons_dat = {'N': 61,
                   'r': r[0].astype(int),
                   'sprefs': sprefs,
                   'c_1': .5,
                   'c_2': .5,
                   'c_50': 13.1,
                   'r_max': r_max,
                   'c_rms': 0.707106781,
                   'sig_tc': 10,
                   'sigtc_sq': sigtc_sq}

    optimal = np.zeros((2, ndata))
    print init, "fo"
    for i in range(len(r)):
        neurons_dat['r'] = r[i].astype(int)
        print init
        op = sm.optimizing(data=neurons_dat, init=init)
        #op = sm.optimizing(data=neurons_dat)
        optimal[0][i], optimal[1][i] = op['s_1'], op['s_2']
        optimal = np.sort(optimal, axis=0)
    return optimal

"""
Multilayer ReLU net
"""

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
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

class COMLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None):
        """
        Layer with Center of Mass decoder
        Params same as above
        """
        self.input = input
        if W is None:
            W_values = (1/np.sqrt(n_in)) * np.random.randn(n_in, n_out)

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W
        
        self.ones = np.ones((n_in, n_out))
        
        self.output = T.dot(input, self.W)/T.dot(input, self.ones)
        
        # parameters of the model
        self.params = [self.W]

class MLP(object):


    def __init__(self, rng, input, n_in, n_hidden, n_out):
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
            #activation=T.nnet.sigmoid
            activation=relu
        )
        
        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden,
            n_out=n_out,
            #activation=relu
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
        return T.mean((self.y_pred[0] - y[0]) ** 2 + (self.y_pred[1] - y[1]) ** 2)
    
    def mse_s1(self, y):
        # error between output and target
        return T.mean((self.y_pred[0] - y[0]) ** 2)
    
    def sym_mse(self, y):
        # error between output and target
        return T.mean(((self.y_pred[0] - y[0]) ** 2 + (self.y_pred[1] - y[1]) ** 2)
                      * ((self.y_pred[1] - y[0]) ** 2 + (self.y_pred[0] - y[1]) ** 2))
        
class COMMLP(object):


    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """
        Params same as above
        """

        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.sigmoid
        )
        
        self.hiddenLayer2 = COMLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden,
            n_out=n_out,
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
        return T.mean((self.y_pred[0] - y[0]) ** 2 + (self.y_pred[1] - y[1]) ** 2)
    
    def sym_mse(self, y):
        # error between output and target
        return T.mean(((self.y_pred[0] - y[0]) ** 2 + (self.y_pred[1] - y[1]) ** 2)
                      * ((self.y_pred[1] - y[0]) ** 2 + (self.y_pred[0] - y[1]) ** 2))
        

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        """
        data_x, data_y, _ = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype='float32'),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype='float32'),
                                 borrow=borrow)
        return shared_x, shared_y

def train_nn(dataset, n_hidden=20, learning_rate=0.01, n_epochs=10, batch_size=20, test_data=None, COM=False, n_in=61, n_out=2):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

   """
    train_set_x, train_set_y = shared_dataset(dataset)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    test_batch_size = 1
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.fmatrix('x')   # input data from visual neurons
    y = T.fmatrix('y')  # posterior

    rng = np.random.RandomState(1234)

    # construct the MLP class
    nn = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    
    if COM == True:
        nn = COMMLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)

    cost = nn.mse(y)

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

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index)

    end_time = time.clock()
    
    return nn, x

def test_nn(nn, nnx, test_data):
    print 'testing'
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
    pred_ys = np.zeros((len(true_ys), 2))
    for i in range(len(true_ys)):
        pred_ys[i] = test_model(i)
        #print test_model(i)[0], true_ys[i]
        print test_model(i)[0] * 90, true_ys[i]
    
    print nn.get_params()
    return pred_ys, true_ys

def plot(nn, optimal, s_1, s_2, ntraindata):
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(nn[0], nn[1], c='b', label='Neural Net')
    ax.scatter(optimal[0], optimal[1], c='r', label='MLE')
    ax.set_xlabel(r'\hat{s_1}',fontsize=16)
    ax.set_ylabel(r'\hat{s_2}',fontsize=16)
    ax.legend()
    name = "{s_1}_{s_2}_{ntraindata}.pdf".format(s_1=s_1, s_2=s_2, ntraindata=ntraindata)
    fig.savefig(name)

def test_models(s_0, s_1, nn, nnx, sm):
    init = {'s_1':s_0,
            's_2':s_1}
    print init
    test_data = generate_s_data(s_0, s_1)
    print test_data
    nn_preds, _ = test_nn(nn, nnx, test_data)
    nn_preds = nn_preds.T * 90
    r, s, c = test_data
    opt_preds = fit_optimal(r, init, sm)
    return nn_preds, opt_preds

def main():
    """
    arguments: [smaller stimulus, larger stimulus, amount of training data]
    """
    s1 = int(sys.argv[1])
    s2 = int(sys.argv[2])
    ntraindata = int(sys.argv[3])

    neurons_code = """
    data {
        int<lower=0> N; // number of neurons
        int r[N]; // neural response
        real sprefs[N]; // preferred stimuli
        real<lower=0> c_1;
        real<lower=0> c_2;
        int r_max;
        //real c_rms;
        //real c_50;
        //real<lower=0> sig_tc;
        real<lower=0> sigtc_sq;
    }
    parameters {
        real s_1;
        real s_2;
    }
    transformed parameters {
        real lambda[N];
        for (n in 1:N)
            // lambda[n] <- r_max * ((c_1 * exp(normal_log(s_1, sprefs[n], sig_tc)) + c_2 * exp(normal_log(s_2, sprefs[n], sig_tc)))/(c_rms + c_50));
            // lambda[n] <- r_max * (c_1 * exp(normal_log(s_1, sprefs[n], sig_tc)) + c_2 * exp(normal_log(s_2, sprefs[n], sig_tc)));
            lambda[n] <- r_max * (c_1 * exp(- square(s_1 - sprefs[n])/(2 * sigtc_sq)) + c_2 * exp(- square(s_2 - sprefs[n])/(2 * sigtc_sq)));
    }
    model {
        s_1 ~ uniform(-60, 60);
        s_2 ~ uniform(-60, 60);
        r ~ poisson(lambda);
    }
    """


    #Setting up models
    sm = pystan.StanModel(model_code=neurons_code)
    #ntraindata = 20000
    train_data = generate_trainset(ntraindata)
    nn, nnx = train_nn(train_data, n_hidden=50, learning_rate=.001, n_epochs=100)

    nn_preds, opt_preds = test_models(s1, s2, nn, nnx, sm)
    plot(nn_preds, opt_preds, s1, s2, ntraindata)

if __name__ == "__main__":
    main()


