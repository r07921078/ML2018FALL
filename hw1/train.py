import csv
import numpy as np 

# Original
ori_data = []

with open('train.csv', 'r', encoding='big5') as train_fin:
	rows = csv.reader(train_fin, delimiter=',')

	is_first = True

	for row in rows:
		if is_first:
			is_first = False
		else:
			tmp = []
			for i in range(3, 27):
				if row[i] == 'NR' or float(row[i]) <= 0:
					tmp.append(float(0))
				else:
					tmp.append(float(row[i]))

			ori_data.append(tmp)


# Processed 18*xxxx
proc_data = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

for i in range(0, len(ori_data), 18):
	for j in range(18):
		proc_data[j].extend(ori_data[i+j])


# remove all zero item
for i in range(1, len(proc_data[0]), 1):
	all_zero = True
	for j in range(18):
		if proc_data[j][i]:
			all_zero = False
			break

	if all_zero:
		for j in range(18):
			proc_data[j][i] = proc_data[j][i-1]

# y = Ax
# x = (A^tA)^-1A^ty for closed form solution
y = []
A = []

# 12 months
hour = 2
#feat = range(18)
feat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13 , 14, 15, 16, 17]

for i in range(12):
	# 24*20-9 = 471 hours
	for j in range(480-hour):
		tmp = [1] + [proc_data[m][n] for m in feat for n in range(i*480+j, i*480+j+hour)]
		A.append(tmp)
		y.append(proc_data[9][i*480+j+hour])



np_y = np.zeros((len(y), 1))
for i in range(len(y)):
	np_y[i] = y[i]



#r_np_y = np.array(y)
#np_y = np.transpose(r_np_y)
np_A = np.array(A)
AtA = np.dot(np_A.T, np_A)
Aty = np.dot(np_A.T, np_y)
i_AtA = np.linalg.inv(AtA)

W = np.dot(i_AtA, Aty) 


np.save('model.npy', W)
