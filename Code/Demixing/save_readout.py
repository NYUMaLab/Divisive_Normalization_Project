import pickle
import os
import numpy as np

posts = {}
testsets = {}
posts_single = {}
testsets_single = {}
for i in range(91):
    file_name = 'output_post/post_' + str(i) + '.pkl'
    if os.path.isfile(file_name):
        pkl_file = open(file_name, 'rb')
        p, r, c, delta_s = pickle.load(pkl_file)
        posts_single[delta_s, c] = p
        testsets_single[delta_s, c] = r

for i in range(61):
    file_name = 'output_post/post_' + str(i) + '.pkl'
    if os.path.isfile(file_name):
        pkl_file = open(file_name, 'rb')
        p, r, c, delta_s = pickle.load(pkl_file)
        if c == [[1], [4]]:
            c = 14
        elif c == [[4], [1]]:
            c = 41
        posts_single[delta_s, c] = p
        testsets_single[delta_s, c] = r

for i in range(31):
    file_name = 'output_post_2/post_2_' + str(i) + '.pkl'
    if os.path.isfile(file_name):
        pkl_file = open(file_name, 'rb')
        p, r, c, delta_s = pickle.load(pkl_file)
        posts[delta_s] = p
        testsets[delta_s] = r

pkl_file = open('readout.pkl', 'wb')
pickle.dump((posts, testsets), pkl_file)
pkl_file.close()

pkl_file = open('readout_single.pkl', 'wb')
pickle.dump((posts_single, testsets_single), pkl_file)
pkl_file.close()