from gensim.models import word2vec as w2v

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, GRU, Activation, Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import sys
import csv
import numpy as np
import jieba as jb

test_x_csv = sys.argv[1]
dict_txt_big = sys.argv[2]
output_csv = sys.argv[3]

np.set_printoptions(suppress=True)
jb.load_userdict(dict_txt_big)
model = w2v.Word2Vec.load('word2vec.model')

## argument
#BATCH_SIZE = 512
#EPOCH = 10
MAX_LENGTH = 250
#INPUT_LENGTH = 250
#MODEL_NAME = 'model9.h5'
BEST_MODEL_NAME = 'models-keras/best14.h5'
#TRAINING_LOG = 'training11.log'
PREDICT = output_csv


pretrained_weights = model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape




word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1

    

sentences = []
with open(test_x_csv, 'r', encoding='utf-8') as f_train_in:
    rows = csv.reader(f_train_in)

    is_first = True

    for row in rows:
        if is_first:
            is_first = False

        else:
            words = jb.cut(row[1], cut_all=True)
            sentences.append(words)
            

            
            
def text_to_index_array(dic, sentences):
    rlt = []
    for sen in sentences:
        l = []
        for word in sen:
            try:
                l.append(dic[word])
            except:
                l.append(0)
        rlt.append(l)

    return np.array(rlt)



X = text_to_index_array(word2idx, sentences)
X = pad_sequences(X, maxlen=MAX_LENGTH)

#==========================================
# 
model = load_model(BEST_MODEL_NAME)

p = model.predict(X)
    
with open(PREDICT, 'wt') as f_test_yout:
    csvout = csv.writer(f_test_yout)
    csvout.writerow(['id', 'label'])

    for i in range(len(p)):
        if p[i] > 0.5:
            csvout.writerow([str(i), str(1)])
        else:
            csvout.writerow([str(i), str(0)])




