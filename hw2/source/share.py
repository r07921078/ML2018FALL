import numpy as np

def sigmoid(x):
	s = np.divide(1.0, (1.0 + np.exp(-x)))
	return np.clip(s, 1e-8, 1-(1e-8))


global feat
feat = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]