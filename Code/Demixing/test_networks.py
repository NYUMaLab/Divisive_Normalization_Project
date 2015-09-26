import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.patches as mpatches
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
    
    c_arr = [[[1, 2, 4], [1, 2, 4]]]
    c = c_arr[i % len(c_arr)]
    perm_cs = dm.cartesian(c)

    file_name = 'output_nn_runs_' + str(j) + '/nn_runs_' + str(j) + '_' + str(i) + '.pkl'
    if os.path.isfile(file_name):
    	pkl_file = open(file_name, 'rb')
    	nn, nnx, valid_mse, stats, c = pickle.load(pkl_file)

    num_deltas = 30
    s1 = -30
    nn_stats = {}
    nn_stats = {'mean_s1': np.zeros((num_deltas, len(perm_cs))), 
                'mean_s2': np.zeros((num_deltas, len(perm_cs))),
                'bias_s1': np.zeros((num_deltas, len(perm_cs))), 
                'bias_s2': np.zeros((num_deltas, len(perm_cs))), 
                'var_s1': np.zeros((num_deltas, len(perm_cs))), 
                'var_s2': np.zeros((num_deltas, len(perm_cs))), 
                'cov': np.zeros((num_deltas, len(perm_cs))), 
                'corr': np.zeros((num_deltas, len(perm_cs))),
                'mse': np.zeros((num_deltas, len(perm_cs))),
                }

    for delta_s in range(num_deltas):
    	for pc in range(len(perm_cs)):
		    test_data = dm.generate_testset(4500, stim_0=s1, stim_1=s1+delta_s, con_0 = perm_cs[pc][0], con_1 = perm_cs[pc][1], r_max=1)
		    nn_preds, _ = dm.test_nn(nn, nnx, test_data)
		    nn_preds = nn_preds.T * 90
		    stats = dm.get_statistics(s1, s1 + delta_s, nn_preds)
		    nn_stats['mean_s1'][delta_s][pc] = stats['mean_s1']
		    nn_stats['mean_s2'][delta_s][pc] = stats['mean_s2']
		    nn_stats['bias_s1'][delta_s][pc] = stats['bias_s1']
		    nn_stats['bias_s2'][delta_s][pc] = stats['bias_s2']
		    nn_stats['var_s1'][delta_s][pc] = stats['var_s1']
		    nn_stats['var_s2'][delta_s][pc] = stats['var_s2']
		    nn_stats['cov'][delta_s][pc] = stats['cov']
		    nn_stats['corr'][delta_s][pc] = stats['corr']
		    nn_stats['mse'][delta_s][pc] = stats['mse']

    file_name = "nn_runs_" + str(j) + "_" + str(i) + ".pkl"

    out = (nn, nnx, valid_mse, nn_stats, c)

    pkl_file = open(file_name, 'wb')
    pickle.dump(out, pkl_file)
    pkl_file.close()

if __name__ == "__main__":
    main()