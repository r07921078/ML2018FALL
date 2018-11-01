import sys
import csv
import numpy as np

import math
import pandas as pandas

from source import share as sh

np.set_printoptions(suppress=True)


train_x = []
train_y = []

with open('train_x.csv', 'r', encoding='big5') as f_train_xin:
	rows = csv.reader(f_train_xin, delimiter=',')

	is_first = True

	for row in rows:
		if is_first:
			is_first = False
		else:
			#train_x.append(row)
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




np_x = np.array(train_x)
np_y = np.array(train_y).astype('int32')

# Normalization
x_mean = np.mean(np_x, axis=0)
x_std = np.std(np_x, axis=0)
np_x = (np_x - x_mean) / x_std

train_data_size = np_x.shape[0]
dim = np_x.shape[1]

# calculate mu1 and mu2
mu1 = np.zeros((dim,))
mu2 = np.zeros((dim,))

tx = np_x.sum(axis=0)
cnt1 = 0
cnt2 = 0

for i in range(train_data_size):
    if np_y[i] == 1:
        mu1 += np_x[i]
        cnt1 += 1
    else:
        mu2 += np_x[i]
        cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

# calculate sigma1 and sigma2
sigma1 = np.zeros((dim,dim))
sigma2 = np.zeros((dim,dim))

for i in range(train_data_size):
    if np_y[i] == 1:
        sigma1 += np.dot(np.transpose([np_x[i] - mu1]), ([np_x[i] - mu1]))
    else :
        sigma2 += np.dot(np.transpose([np_x[i] - mu2]), ([np_x[i] - mu2]))

sigma1 /= cnt1
sigma2 /= cnt2

shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2

# 存model
param = {'mu1':mu1, 'mu2':mu2, 'N1':cnt1, 'N2':cnt2, 'shared_sigma':shared_sigma}
np.save('g_model.npy', param)
