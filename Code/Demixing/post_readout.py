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

	pkl_file = open('readout.pkl', 'rb')
	posts, testsets = pickle.load(pkl_file)

	file_name = 'output_nn_tests_' + str(j) + '/nn_tests_' + str(j) + '_' + str(i) + '.pkl'
	if os.path.isfile(file_name):
	    pkl_file = open(file_name, 'rb')
	    nn, nnx, valid_mse, _, _ = pickle.load(pkl_file)

	x = []
	y = []
	for s_i in range(31):
	    x.append(dm.get_hu_responses(testsets[s_i], nn))
	    y.append(np.array((1/posts[s_i]['var_s1'], 1/posts[s_i]['var_s2'])))
	y = np.concatenate(y, axis=1).T
	x = np.concatenate(x)
	inds = range(len(x))
	np.random.shuffle(inds)
	x_shuf = x[inds]
	y_shuf = y[inds]
	validset_hid = x_shuf[0:2000], y_shuf[0:2000]
	trainset_hid = x_shuf[2000:], y_shuf[2000:]

	r = []
	for s_i in range(31):
	    r.append(testsets[s_i])
	r = np.concatenate(r)
	r_shuf = r[inds]
	validset_r = r_shuf[0:2000], y_shuf[0:2000]
	trainset_r = r_shuf[2000:], y_shuf[2000:]

	nn_lin, nnx_lin, valid_mse_lin = dm.train_nn(trainset_hid, valid_dataset=validset_hid, n_in=20, learning_rate=0.0001, n_epochs=100, linear=True, rho=.9, mu=.99, nesterov=True)
	nn_full, nnx_full, valid_mse_full = dm.train_nn(trainset_r, valid_dataset=validset_r, learning_rate=0.0001, mult_ys=False, n_epochs=100, rho=.9, mu=.99, nesterov=True)

	hus, vpost = validset_hid
	lin_preds = dm.get_hu_responses(hus, nn_lin)
	full_preds, _ = dm.test_nn(nn_full, nnx_full, validset_r)
	sum_preds = np.sum(hus, axis=1)
	vp = np.concatenate((vpost.T[0], vpost.T[1]))
	lin_preds = np.concatenate((lin_preds.T[0], lin_preds.T[1]))
	full_preds = np.concatenate((full_preds.T[0], full_preds.T[1]))
	sum_preds = np.concatenate((sum_preds, sum_preds))

	lin_corr = np.corrcoef(vp, lin_preds)
	full_corr = np.corrcoef(vp, full_preds)
	sum_corr = np.corrcoef(vp, sum_preds)

	pkl_file = open('readout_' + str(j) + '_' + str(i) + '.pkl', 'wb')
	pickle.dump((lin_preds, lin_corr, full_preds, full_corr, sum_preds, sum_corr, valid_mse_lin, valid_mse_full, vp), pkl_file)
	pkl_file.close()
if __name__ == "__main__":
	main()