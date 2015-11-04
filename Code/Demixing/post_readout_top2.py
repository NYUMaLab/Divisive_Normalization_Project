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

	file_name = 'Minicon_Outputs/output_nn_tests_' + str(j) + '/nn_tests_' + str(j) + '_' + str(i) + '.pkl'
	if os.path.isfile(file_name):
	    pkl_file = open(file_name, 'rb')
	    nn, nnx, valid_mse, _, _ = pickle.load(pkl_file)
	c_arr = [[[1, 2, 4], [1, 2, 4]]]
	s_arr = np.linspace(-60, 60, 120)
	acts = dm.get_mean_acts(s_arr, nn, c_arr[0])

	posts = {}
	testsets = {}
	for s_i in range(31):
	    file_name = 'Minicon_Outputs/output_post_2/post_2_' + str(s_i) + '.pkl'
	    if os.path.isfile(file_name):
	        pkl_file = open(file_name, 'rb')
	        p, r, c, delta_s = pickle.load(pkl_file)
	        posts[delta_s] = p
	        testsets[delta_s] = r

	x_hus = []
	x_ins = []
	y = []
	for s_i in range(31):
	    x_hus.append(dm.get_hu_responses(testsets[s_i], nn))
	    x_ins.append(testsets[s_i])
	    y.append(np.array((1/posts[s_i]['var_s1'], 1/posts[s_i]['var_s2'])))
	y = np.concatenate(y, axis=1).T
	x_hus = np.concatenate(x_hus)
	x_ins = np.concatenate(x_ins)
	inds = range(len(x_hus))
	np.random.shuffle(inds)
	x_hus_shuf = x_hus[inds]
	x_ins_shuf = x_ins[inds]
	y_shuf = y[inds]
	validset_hus = x_hus_shuf[0:2000], y_shuf[0:2000]
	validset_ins = x_ins_shuf[0:2000], y_shuf[0:2000]
	trainset_hus = x_hus_shuf[2000:], y_shuf[2000:]
	trainset_ins = x_ins_shuf[2000:], y_shuf[2000:]    

	int_acts = [np.sum(acts[:, :, hu_i]) for hu_i in range(20)]

	sort_acts = np.argsort(int_acts)

	trainset_top2 = (trainset_hus[0][:, sort_acts[0:1]], trainset_hus[1])
	validset_top2 = (validset_hus[0][:, sort_acts[0:1]], validset_hus[1])

	weights_hus = np.linalg.lstsq(trainset_hus[0], trainset_hus[1])[0]
	weights_top2 = np.linalg.lstsq(trainset_top2[0], trainset_top2[1])[0]
	weights_ins = np.linalg.lstsq(trainset_ins[0], trainset_ins[1])[0]

	hus, vpost = validset_hus
	hus_top2, _ = validset_top2
	ins, _ = validset_ins
	lin_preds_hus = np.dot(hus, weights_hus)
	lin_preds_top2 = np.dot(hus_top2, weights_top2)
	lin_preds_ins = np.dot(ins, weights_ins)
	vp = np.concatenate((vpost.T[0], vpost.T[1]))
	lin_preds_corr_hus = np.concatenate((lin_preds_hus.T[0], lin_preds_hus.T[1]))
	lin_preds_corr_top2 = np.concatenate((lin_preds_top2.T[0], lin_preds_top2.T[1]))
	lin_preds_corr_ins = np.concatenate((lin_preds_ins.T[0], lin_preds_ins.T[1]))

	lin_corr_hus = np.corrcoef(vp, lin_preds_corr_hus)[0, 1]
	lin_corr_top2 = np.corrcoef(vp, lin_preds_corr_top2)[0, 1]
	lin_corr_ins = np.corrcoef(vp, lin_preds_corr_ins)[0, 1]

	file_name = "post_readoutt2_" + str(j) + "_" + str(i) + ".pkl"

	out = (lin_corr_hus, lin_corr_top2, lin_corr_ins, lin_preds_hus, lin_preds_top2, lin_preds_ins, weights_hus, weights_top2, weights_ins, sort_acts)

	pkl_file = open(file_name, 'wb')
	pickle.dump(out, pkl_file)
	pkl_file.close()

if __name__ == "__main__":
    main()

