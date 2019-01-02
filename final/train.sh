#!/bin/bash

#python train.py Train_folder Train_csv Model_name
python train.py /mnt/e/ML_dataset/final/Train /mnt/e/ML_dataset/final/train.csv myModel.h5
#python set_thred.py Train_folder Train_csv Model_name 
python set_thred.py /mnt/e/ML_dataset/final/Train /mnt/e/ML_dataset/final/train.csv myModel.h5
