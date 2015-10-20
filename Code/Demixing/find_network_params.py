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
from scipy.stats import kurtosis

def main():    
    i = int(sys.argv[1])

    lrs = np.array([.01, .005, .001, .0005, .0001, .00001])
    rhos = np.array([0, .75, .9, .95, .99, .999])
    mus = np.array([0, .75, .9, .95, .99, .999])
    nests = np.array([True, False])
    c = [[1, 2, 4], [1, 2, 4]]
    tds = [4500, 9000, 45000, 90000, 450000, 900000]
    hus = [10, 20, 50, 100, 200, 500, 1000]
    params = dm.cartesian([lrs, rhos, mus, nests, tds, hus])
    lr, rho, mu, n, td, hu = params[i]

    train_data = dm.generate_trainset(td, discrete_c=c, r_max=1)
    valid_data = dm.generate_testset(900, discrete_c=c, r_max=1)
    nn, nnx, valid_mse = dm.train_nn(train_data, valid_dataset=valid_data, n_hidden=hu, learning_rate=lr, n_epochs=100, rho=rho, mu=mu, nesterov=n)

    file_name = "nn_optim" + str(i) + ".pkl"

    out = (nn, nnx, valid_mse, c, lr, rho, mu, n, td, hu)

    pkl_file = open(file_name, 'wb')
    pickle.dump(out, pkl_file)
    pkl_file.close()

if __name__ == "__main__":
    main()