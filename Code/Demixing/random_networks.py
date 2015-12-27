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

	nneuron = 61
	n_hus = 1000

	nn_file_name = 'output_nn_runs_2/nn_runs_2_' + str(i) + '.pkl'
	if os.path.isfile(nn_file_name):
	    pkl_file = open(nn_file_name, 'rb')
	    nn, nnx, valid_mse, _, _ = pickle.load(pkl_file)

	posts_v1 = {}
	posts_v2 = {}
	testsets = {}
	for s_i in range(90):
	    post_file_name = 'output_post_4/post_4_' + str(s_i) + '.pkl'
	    if os.path.isfile(post_file_name):
	        pkl_file = open(post_file_name, 'rb')
	        p, r, c, s, delta_s = pickle.load(pkl_file)
	        tc_i = tuple(c[i])
	        if tc_i in posts_v1:
	            posts_v1[tc_i] = np.append(posts_v1[tc_i], p['var_s1'])
	            posts_v2[tc_i] = np.append(posts_v2[tc_i], p['var_s2'])
	            testsets[tc_i] = np.append(testsets[tc_i], r, axis = 0)
	        else:
	            posts_v1[tc_i] = p['var_s1']
	            posts_v2[tc_i] = p['var_s2']
	            testsets[tc_i] = r

	rand_nn = {}
	nn_params = nn.get_params()
	b = nn_params['b']
	W = nn_params['W']
	means_W = np.mean(W)
	vars_W = np.var(W)
	means_b = np.mean(b)
	vars_b = np.var(b)
	rand_nn['W'] = vars_W * np.random.randn(nneuron, n_hus) + means_W
	rand_nn['b'] = vars_b * np.random.randn(n_hus) + means_b

	lci, lch, lcio, lcho, kc, sc, lcia, lcha, kca, sca = dm.get_corr(rand_nn, testsets, posts_v1, posts_v2, rand_nn=True)

	out = (nn, lci, lch, lcio, lcho, kc, sc, lcia, lcha, kca, sca)

	out_file_name = 'rand_nets_' + str(i) + '.pkl'
	pkl_file = open(out_file_name, 'wb')
	pickle.dump(out, pkl_file)
	pkl_file.close()

if __name__ == "__main__":
	main()
