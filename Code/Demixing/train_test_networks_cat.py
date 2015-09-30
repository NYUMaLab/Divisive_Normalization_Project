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
 
    td = dm.generate_trainset_cat(20000, low=1, high=4, r_max=1)
    vd = dm.generate_trainset_cat(3000, low=1, high=4, r_max=1)
    nn, nnx, valid_mse= train_nn(td_cat, valid_dataset=vd_cat, n_hidden=20, learning_rate=.0005, n_epochs=100, rho=.9, n_out=1)

    num_deltas = 30
    s2 = -30
    nn_stats = {'mean': np.zeros(num_deltas),  
                'bias': np.zeros(num_deltas), 
                'var': np.zeros(num_deltas), 
                'mse': np.zeros(num_deltas),
                }

    for delta_s in range(num_deltas):
        test_data = generate_testset(4500, stim_0=s2+delta_s, stim_1=s2, discrete_c=c, r_max=1)
        nn_preds, _ = test_nn(nn, nnx, test_data)
        nn_preds = nn_preds.T * 90
        stats = get_statistics_cat(s2_delta_s, nn_preds)
        nn_stats['mean'][delta_s] = stats['mean']
        nn_stats['bias'][delta_s] = stats['bias']
        nn_stats['var'][delta_s] = stats['var']
        nn_stats['mse'][delta_s] = stats['mse']

    file_name = "nn_runs_cat_" + str(i) + ".pkl"

    out = (nn, nnx, valid_mse, nn_stats)

    pkl_file = open(file_name, 'wb')
    pickle.dump(out, pkl_file)
    pkl_file.close()

if __name__ == "__main__":
    main()