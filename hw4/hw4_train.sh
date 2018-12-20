#!/bin/bash
cat word2vec.model.* > word2vec.model
python3 word2vec_mod.py $1 $2 $3 $4
python3 train.py $1 $2 $3 $4
