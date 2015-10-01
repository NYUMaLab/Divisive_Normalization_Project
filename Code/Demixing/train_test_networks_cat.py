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
 
    td = dm.generate_trainset_cat(270000, c_0=4, c_1=1, r_max=1)
    vd = dm.generate_trainset_cat(900, c_0=4, c_1=1, r_max=1)
    nn, nnx, valid_mse = dm.train_nn(td, valid_dataset=vd, n_hidden=20, learning_rate=0.0001, n_epochs=100, rho=.9, mu=.99, nesterov=True, n_out=1)

    num_deltas = 30
    s2 = -30
    nn_stats = {'mean': np.zeros(num_deltas),  
                'bias': np.zeros(num_deltas), 
                'var': np.zeros(num_deltas), 
                'mse': np.zeros(num_deltas),
                }

    for delta_s in range(num_deltas):
        test_data = dm.generate_testset_cat(4500, s2+delta_s, s2, r_max=1)
        nn_preds, _ = dm.test_nn(nn, nnx, test_data)
        nn_preds = nn_preds.T * 90
        stats = dm.get_statistics_cat(s2+delta_s, nn_preds)
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