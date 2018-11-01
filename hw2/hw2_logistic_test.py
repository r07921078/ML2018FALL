import sys
import csv
import numpy as np

import math
import pandas as pandas

from source import share as sh

np.set_printoptions(suppress=True)

infile = sys.argv[1]
outfile = sys.argv[2]

w = np.load('l_model.npy')

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


# normalization
test_x = np.array(test_x)
x_mean = np.mean(test_x, axis=0)
x_std = np.std(test_x, axis=0)
test_x = (test_x - x_mean) / x_std

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)

z = sh.sigmoid(np.dot(test_x, w))

with open(outfile, 'wt') as f_test_yout:
	csvout = csv.writer(f_test_yout)
	csvout.writerow(['id', 'value'])

	for i in range(len(z)):
		if z[i] > 0.5:
			csvout.writerow(['id_'+str(i), '1'])
		else:
			csvout.writerow(['id_'+str(i), '0'])


