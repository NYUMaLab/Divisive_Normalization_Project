import numpy as np
import math
import os
import sys
import time
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

def generate_s_data(stim_0, stim_1):
    ndata = 3000
    c_0, c_1 = np.ones((2, ndata)) * .5
    s_0, s_1 = np.ones((2, ndata))
    s_0 = s_0 * stim_0
    s_1 = s_1 * stim_1
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
    
if __name__ == "__main__":
    main()