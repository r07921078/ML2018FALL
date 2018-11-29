import csv
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, LeakyReLU, PReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import History ,ModelCheckpoint
from keras.layers.normalization import BatchNormalization


infile = sys.argv[1]
outfile = sys.argv[2]


def normalization(x):
    x = (x - x.mean()) / x.std()
    return x


test_x = []


with open(infile, 'r', encoding='big5') as f_test_in:
    rows = csv.reader(f_test_in)
    is_first = True

    for row in rows:
        if is_first:
            is_first = False

        else:
            test_x += row[1].strip().split(' ')
            
            

#test_x = np.array(test_x).reshape(-1, 48, 48, 1)
test_x = np.array(test_x, dtype=float).reshape(-1, 48, 48, 1)
test_x = normalization(test_x)

# public and private model are the same
model = load_model("model.h5")

p = model.predict(test_x)

pred_y = []
for i in p:
    print(i)
    pred_y.append(np.argmax(i))

    
with open(outfile, 'wt') as f_test_yout:
    csvout = csv.writer(f_test_yout)
    csvout.writerow(['id', 'label'])

    for i in range(len(pred_y)):
        csvout.writerow([str(i), str(pred_y[i])])
        
    
        
