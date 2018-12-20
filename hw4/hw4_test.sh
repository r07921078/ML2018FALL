#!/bin/bash
cat word2vec.model.* > word2vec.model
python3 test.py $1 $2 $3
