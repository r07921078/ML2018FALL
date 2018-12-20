from gensim.models import word2vec as w2v
import sys
import csv
import numpy as np
import jieba as jb

train_x_csv = sys.argv[1]
train_y_csv = sys.argv[2]
text_x_csv = sys.argv[3]
dict_txt_big = sys.argv[4]


jb.load_userdict(dict_txt_big)

# generate word2vec model
outfile = open('tmp.csv', 'w', encoding='utf-8')
with open(train_x_csv, 'r', encoding='utf-8') as f_train_in:
    rows = csv.reader(f_train_in)

    is_first = True

    for row in rows:
        if is_first:
            is_first = False

        else:
            words = jb.cut(row[1], cut_all=True)
            for word in words:
                outfile.write(word + ' ')
            outfile.write('\n')

#加入testing data作為 word2vec training
with open(text_x_csv, 'r', encoding='utf-8') as f_train_in:
    rows = csv.reader(f_train_in)

    is_first = True

    for row in rows:
        if is_first:
            is_first = False

        else:
            words = jb.cut(row[1], cut_all=True)
            for word in words:
                outfile.write(word + ' ')
            outfile.write('\n')
            
outfile.close()

sentences = w2v.LineSentence('tmp.csv')
model = w2v.Word2Vec(sentences, size=250, sg=1)
model.save("word2vec.model")


