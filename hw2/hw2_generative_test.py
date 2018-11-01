import sys
import csv
import numpy as np

import math
import pandas as pandas

from source import share as sh

np.set_printoptions(suppress=True)

def predict(test, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inverse)
    x = np.transpose(test)
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1) / N2)
    a = np.dot(w, x) + b
    y = sh.sigmoid(a)
    return y


infile = sys.argv[1]
outfile = sys.argv[2]


param = np.load('g_model.npy')
mu1 = param.item().get('mu1')
mu2 = param.item().get('mu2')
cnt1 = param.item().get('N1')
cnt2 = param.item().get('N2')
shared_sigma = param.item().get('shared_sigma')

test_x = []

with open(infile, 'r', encoding='big5') as f_test_xin:
	rows = csv.reader(f_test_xin, delimiter=',')

	is_first = True


	for row in rows:
		if is_first:
			is_first = False
		else:
			select_feat = []
			for i in sh.feat:
				select_feat.append(float(row[i]))

			test_x.append(select_feat)
 
test_x = np.array(test_x)

# Normalization
x_mean = np.mean(test_x, axis=0)
x_std = np.std(test_x, axis=0)
test_x = (test_x - x_mean) / x_std

z = predict(test_x, mu1, mu2, shared_sigma, cnt1, cnt2)

with open(outfile, 'wt') as f_test_yout:
	csvout = csv.writer(f_test_yout)
	csvout.writerow(['id', 'value'])

	for i in range(len(z)):
		if z[i] > 0.5:
			csvout.writerow(['id_'+str(i), '1'])
		else:
			csvout.writerow(['id_'+str(i), '0'])

