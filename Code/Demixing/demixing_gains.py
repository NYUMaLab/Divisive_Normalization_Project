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
from itertools import product
from scipy.stats import poisson
import matplotlib.patches as mpatches

nneuron = 61
min_angle = -90
max_angle = 90
sprefs = np.linspace(min_angle, max_angle, nneuron)
eps = np.finfo(np.float64).eps
np.random.seed(1234)

r_max = 10
sigtc_sq = float(10**2)
sigtc = 10
c_50 = 13.1

def random_c2d(ndata, low, high, sort):
    c_range = high - low
    c = np.random.rand(2, ndata) * c_range + low
    if sort:
        c = np.sort(c, axis=0)
    return c[0], c[1]

def random_c(ndata, ndims, low, high, sort):
    c_range = high - low
    if ndims == 1:
        c = np.random.rand(ndims, ndata)[0] * c_range + low
    c = np.random.rand(ndims, ndata) * c_range + low
    if sort:
        c = np.sort(c, axis=0)
    return c

def generate_trainset(ndata, low=.3, high=1.3, crange=.5, r_max=10):
    s_0, s_1 = np.random.rand(2, ndata) * 120 - 60
    c_0 = random_c(ndata, 1, low, low+crange, True)
    c_1 = random_c(ndata, 1, high, high+crange, True)
    r, s, c = generate_popcode_data(ndata, nneuron, sigtc_sq, c_50, r_max, "poisson", True, s_0, s_1, c_0, c_1)
    return r, s, c

def generate_testset(ndata, low=.3, high=1.3, r_max=10):
    c_0, c_1 = random_c2d(ndata, low, high, True)
    s_0, s_1 = np.random.rand(2, ndata) * 120 - 60
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

def get_statistics(s1, s2, preds):
    mean_s1 = np.mean(preds[0])
    mean_s2 = np.mean(preds[1])
    bias_s1 = mean_s1 - s1
    bias_s2 = mean_s2 - s2
    covmat = np.cov(preds)
    var_s1 = covmat[0, 0]
    var_s2 = covmat[1, 1]
    cov = covmat[0, 1]
    corr = cov / (np.sqrt(var_s1) * np.sqrt(var_s2))
    stats = {'mean_s1': mean_s1, 'mean_s2': mean_s2, 'bias_s1': bias_s1, 'bias_s2': bias_s2, 'var_s1': var_s1, 'var_s2': var_s2, 'cov': cov, 'corr': corr}
    return stats

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
    
    def valid_mse(self, y):
        return T.mean(((self.y_pred[0] * 90) - (y[0] * 90)) ** 2 + ((self.y_pred[1] * 90) - (y[1] * 90)) ** 2)
    
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

def train_nn(train_dataset, valid_dataset=None, n_hidden=20, learning_rate=0.01, n_epochs=10, batch_size=20, test_data=None, COM=False, rho=False, nesterov=True, momentum=0, n_in=61, n_out=2):
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
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.fmatrix('x')   # input data from visual neurons
    y = T.fmatrix('y')  # ground truth

    rng = np.random.RandomState(1234)

    # construct the MLP class
    nn = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    
    if COM:
        nn = COMMLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
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
        updates = RMSprop(cost, nn.params, rho=rho, learning_rate=learning_rate, mu=momentum, nesterov=nesterov)
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
    
    validate_model = theano.function(
        inputs=[index],
        outputs=nn.valid_mse(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    start_time = time.clock()

    epoch = 0 
    done_looping = False
    
    if valid_dataset:
        valid_mse = np.zeros(n_epochs)

    while (epoch < n_epochs) and (not done_looping):
        for minibatch_index in xrange(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index)
            
        if valid_dataset:
            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            valid_mse[epoch] = this_validation_loss 

            print(
                'epoch %i, minibatch %i/%i, validation error %f' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss
                )
            )
            
        epoch = epoch + 1

    end_time = time.clock()
    
    if valid_dataset:
        return nn, x, valid_mse
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
        #print test_model(i)[0] * 90, true_ys[i]
    
    #print nn.get_params()
    return pred_ys, true_ys

def main():
	index = int(sys.argv[1])

	rhos = np.linspace(.9, .999, 20)
	moms = np.linspace(.9, .999, 20)
	lrs = np.linspace(.00001, .1, 20)
	params = list(product(rhos, moms, lrs))

	rho = params[index][0]
	mom = params[index][1]
	lr = params[index][2]

    train_data_1 = generate_trainset(500000)
	valid_data_1 = generate_testset(3000)
    nn, nnx, valid_mse = train_nn(train_data_1, valid_dataset=valid_data_1, n_hidden=20, learning_rate=lr, n_epochs=100, rho=rho, momentum=mom)

    print valid_mse[99]
if __name__ == "__main__":
    main()


