import sys
import csv
import numpy as np

import math
import pandas as pandas

from source import share as sh

np.set_printoptions(suppress=True)

def logistic_regression(x, y, lr, iteration):
    x_t = x.transpose()
    w = np.zeros(len(x[0]))
    s_gra = np.zeros(len(x[0]))
    for i in range(iteration):
        z = np.dot(x, w)
        prob = sh.sigmoid(z)
        gra = -np.dot(x_t, y-prob) + 2 * w
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        w = w - lr * gra/ada
    return w

# Original
train_x = []
train_y = []

with open('train_x.csv', 'r', encoding='big5') as f_train_xin:
	rows = csv.reader(f_train_xin, delimiter=',')

	is_first = True

	for row in rows:
		if is_first:
			is_first = False
		else:
			select_feat = []
			for i in sh.feat:
				select_feat.append(float(row[i]))

			train_x.append(select_feat)


with open('train_y.csv', 'r', encoding='big5') as f_train_yin:
	rows = csv.reader(f_train_yin, delimiter=',')

	is_first = True

	for row in rows:
		if is_first:
			is_first = False
		else:
			train_y.append(row[0])

# x : 20000*23
# y : 20000*1
np_x = np.array(train_x, dtype=float)
np_y = np.array(train_y, dtype=float)


# normalization
x_mean = np.mean(np_x, axis=0)
x_std = np.std(np_x, axis=0)
np_x = (np_x - x_mean) / x_std

# add bias
np_x = np.concatenate((np.ones((np_x.shape[0], 1)), np_x), axis=1)

iteration = 30000
lr = 0.1

w = logistic_regression(np_x, np_y, lr, iteration)

np.save('l_model.npy', w)
