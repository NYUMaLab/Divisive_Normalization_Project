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
    delta_s = int(sys.argv[1])

    s2=-30
    post_func = dm.posterior_setup_cat(high=4, low=1, num_s=120, r_max=1)
    test_data = dm.generate_testset_cat(50, s2+delta_s, s2, r_max=1)
    r, y, s, c, numvec = test_data
    posts = dm.get_posteriors_pool_cat(r, post_func)
    output = (posts, r, delta_s)

    file_name = "postcat_" + str(delta_s) + ".pkl"

    pkl_file = open(file_name, 'wb')
    pickle.dump(output, pkl_file)
    pkl_file.close()

if __name__ == "__main__":
    main()