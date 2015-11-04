import numpy as np
import theano
import theano.tensor as T
from functools import partial
import pandas as pd
import pickle
import time
import sys
import os
import multiprocessing as mp
import demixing as dm
from demixing import MLP, HiddenLayer

def main():
    i = int(sys.argv[1])
    j = int(sys.argv[2])

    
    if j == 1:
        #c_arr = [[[1], [1]], [[2], [2]], [[4], [4]], [[1], [4]], [[4], [1]]]
        c_arr = [[[1], [4]], [[4], [1]]]
        tc_arr = [[[1], [4]], [[4], [1]]]
        tc = tc_arr[i % len(c_arr)]
        c = c_arr[i % len(c_arr)]
        train_data = dm.generate_trainset(270000, discrete_c=tc, r_max=1)
    elif j == 2:
        c_arr = [[[1, 2, 4], [1, 2, 4]]]
        tc_arr = [[[1, 2, 4], [1, 2, 4]]]
        tc = tc_arr[i % len(c_arr)]
        c = c_arr[i % len(c_arr)]
        train_data = dm.generate_trainset(270000, discrete_c=tc, r_max=1)
    elif j == 3:
        c_arr = [[[1, 2, 4], [1, 2, 4]]]
        tc_arr = [[[1, 2, 4], [1, 2, 4]]]
        tc = tc_arr[i % len(c_arr)]
        c = c_arr[i % len(c_arr)]
        train_data = dm.generate_trainset(270000, highlow = True, discrete_c=tc, r_max=1)
    
    valid_data = dm.generate_testset(900, discrete_c=c, r_max=1)
    #nn, nnx, valid_mse = dm.train_nn(train_data, n_hidden=100, valid_dataset=valid_data, learning_rate=0.0001, n_epochs=100, rho=.9, mu=.99, nesterov=True)
    nn, nnx, valid_mse = dm.train_nn(train_data, n_hidden=100, valid_dataset=valid_data, learning_rate=0.0001, n_epochs=100, rho=.9, mu=.99, nesterov=True, COM=True)

    num_deltas = 30
    s1 = -30
    nn_stats = {'mean_s1': np.zeros(num_deltas), 
                'mean_s2': np.zeros(num_deltas), 
                'bias_s1': np.zeros(num_deltas), 
                'bias_s2': np.zeros(num_deltas), 
                'var_s1': np.zeros(num_deltas), 
                'var_s2': np.zeros(num_deltas), 
                'cov': np.zeros(num_deltas), 
                'corr': np.zeros(num_deltas),
                'mse': np.zeros(num_deltas),
                }

    for delta_s in range(num_deltas):
        test_data = dm.generate_testset(4500, stim_0=s1, stim_1=s1+delta_s, discrete_c=c, r_max=1)
        nn_preds, _ = dm.test_nn(nn, nnx, test_data)
        nn_preds = nn_preds.T * 90
        stats = dm.get_statistics(s1, s1 + delta_s, nn_preds)
        nn_stats['mean_s1'][delta_s] = stats['mean_s1']
        nn_stats['mean_s2'][delta_s] = stats['mean_s2']
        nn_stats['bias_s1'][delta_s] = stats['bias_s1']
        nn_stats['bias_s2'][delta_s] = stats['bias_s2']
        nn_stats['var_s1'][delta_s] = stats['var_s1']
        nn_stats['var_s2'][delta_s] = stats['var_s2']
        nn_stats['cov'][delta_s] = stats['cov']
        nn_stats['corr'][delta_s] = stats['corr']
        nn_stats['mse'][delta_s] = stats['mse']

    file_name = "nn_runs_" + str(j) + "_" + str(i) + ".pkl"

    out = (nn, nnx, valid_mse, nn_stats, c)

    pkl_file = open(file_name, 'wb')
    pickle.dump(out, pkl_file)
    pkl_file.close()

if __name__ == "__main__":
    main()