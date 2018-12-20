from gensim.models import word2vec as w2v
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, GRU, Activation, Bidirectional, Flatten, TimeDistributed
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import sys
import csv
import numpy as np
import jieba as jb

train_x_csv = sys.argv[1]
train_y_csv = sys.argv[2]
text_x_csv = sys.argv[3]
dict_txt_big = sys.argv[4]


np.set_printoptions(suppress=True)
jb.load_userdict(dict_txt_big)
model = w2v.Word2Vec.load('word2vec.model')

## argument
BATCH_SIZE = 512
EPOCH = 50
MAX_LENGTH = 250
INPUT_LENGTH = 250
MODEL_NAME = 'model14.h5'
BEST_MODEL_NAME = 'models-keras/best14.h5'
TRAINING_LOG = 'training14.log'


pretrained_weights = model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape


word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size), dtype=np.float)
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]


print(embeddings_matrix.shape)
print(embeddings_matrix)



############################################################################
# read train_y
############################################################################
train_y = [None for _ in range(120000)]
with open(train_y_csv, 'r', encoding='utf-8') as f_train_in:
    rows = csv.reader(f_train_in)

    is_first = True

    for row in rows:
        if is_first:
            is_first = False

        else:
            train_y[int(row[0])] = float(row[1])


Y = np.array(train_y)
print(Y.shape)

############################################################################
# read train_x
############################################################################
train_x = []
with open(train_x_csv, 'r', encoding='utf-8') as f_train_in:
    rows = csv.reader(f_train_in)

    is_first = True

    for row in rows:
        if is_first:
            is_first = False

        else:
            words = jb.cut(row[1], cut_all=True)
            train_x.append(words)

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



X = text_to_index_array(word2idx, train_x)
X = pad_sequences(X, maxlen=MAX_LENGTH)
print(X.shape)
print("X[0] = ")
print(X[0])





# Build Model
model = Sequential()

# using pretrain weights
model.add(Embedding(output_dim=emdedding_size, 
                    input_dim=vocab_size + 1, mask_zero=True, 
                    weights=[embeddings_matrix], 
                    input_length=INPUT_LENGTH, 
                    trainable=False))

model.add(Bidirectional(GRU(units=128, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
model.add(GRU(units=256, activation='relu', recurrent_activation='hard_sigmoid', dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
model.compile(loss='binary_crossentropy', 
              optimizer=adam, 
              metrics=['accuracy'])

model.summary()

# Setting callback functions
csv_logger = CSVLogger(TRAINING_LOG)
checkpoint = ModelCheckpoint(filepath=BEST_MODEL_NAME,
                             verbose=1,
                             save_best_only=True,
                             monitor='val_acc',
                             mode='max')
earlystopping = EarlyStopping(monitor='val_acc', 
                              patience=6, 
                              verbose=1, 
                              mode='max')


model.fit(X, Y, 
          batch_size=BATCH_SIZE, 
          epochs=EPOCH, 
          validation_split=0.1, 
          callbacks=[earlystopping, checkpoint, csv_logger])



# svae model
model.save(MODEL_NAME)








