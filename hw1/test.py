import sys
import csv
import numpy as np

infile = sys.argv[1]
outfile = sys.argv[2]

# 12 months
hour = 2
#feat = range(18)
feat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17]


W = np.load('model.npy')

x = [W[i][0] for i in range(W.shape[0])] # x is paramater



########## testing ##########
# Original test data
ori_test_data = []

with open(infile, 'r', encoding='big5') as test_fin:
	test_rows = csv.reader(test_fin, delimiter=',')


	for row in test_rows:
		tmp = []
		# 
		for i in range(11-hour, 11):
			if row[i] == 'NR' or float(row[i]) < 0:
				tmp.append(float(0))
			else:
				tmp.append(float(row[i]))

		ori_test_data.append(tmp)


result = [['id_'+str(i)] for i in range(260)]

for i in range(len(result)):
	feature = [1]
	for j in feat:
		feature.extend(ori_test_data[i*18+j])
	
	PM2dot5 = sum([a*b for a,b in zip(x, feature)])
	result[i].append(PM2dot5)

result.insert(0, ['id', 'value'])

with open(outfile, 'wt') as fout:
	csvout = csv.writer(fout)
	csvout.writerows(result)
