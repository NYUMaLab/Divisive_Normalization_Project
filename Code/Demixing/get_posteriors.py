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
    """
    arguments: [smaller stimulus, larger stimulus, amount of training data]
    """
    s_i = int(sys.argv[1])
    j = int(sys.argv[2])
    
    if j == 1:
        c_arr = [[[1], [1]], [[2], [2]], [[4], [4]], [[1], [4]], [[4], [1]]]
    else:
        c_arr = [[[1, 2, 4], [1, 2, 4]]]
        
    print c_arr
    delta_s = s_i / len(c_arr)
    i = s_i % len(c_arr)
    c = c_arr[i]

    post_func = dm.posterior_setup(discrete_c=c, num_s=60, r_max=1)
    if j == 3:
    	test_data = dm.generate_trainset(4500, discrete_c=c, r_max=1)
    else:
	    s1 = -30
	    test_data = dm.generate_testset(4500, stim_0=s1, stim_1=s1+delta_s, discrete_c=c, r_max=1)
    r, test_ss, test_cs = test_data
    posts = dm.get_posteriors_pool(r, post_func)
    output = (posts, r, test_cs, test_ss, delta_s)

    file_name = "post_" + str(j) + "_" + str(s_i) + ".pkl"

    pkl_file = open(file_name, 'wb')
    pickle.dump(output, pkl_file)
    pkl_file.close()

if __name__ == "__main__":
    main()