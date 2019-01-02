import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm

import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras



class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        #assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        R = np.array(Image.open(path+'_red.png'))
        G = np.array(Image.open(path+'_green.png'))
        B = np.array(Image.open(path+'_blue.png'))
        Y = np.array(Image.open(path+'_yellow.png'))

        #image = np.stack((
        #    R/2 + Y/2, 
        #    G/2 + Y/2, 
        #    B),-1)
        image = np.stack((R,G,B,Y),-1)

        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image  
                
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    


class GetThreshold():

    def __init__(self, model_name):
        if model_name is not None:
            self.model = load_model(model_name, custom_objects={'f1': GetThreshold.f1})

    def GetY(self,csvpath,n_class=28):
        self.DA = pd.read_csv(csvpath)
        pre_Y = self.DA['Target'].str.split().tolist()
        Y = np.zeros((len(pre_Y),n_class))
        for cc,target in enumerate(pre_Y):
            target = [int(i) for i in target]
            Y[cc,target] = 1
        self.Y = Y

    def GetScore(self, Train_path, INPUT_SHAPE = (299,299,4), save_name = None):
        MM = np.zeros((self.DA.shape[0],28))
        cc = 0
        for name in tqdm(self.DA['Id']):
            path = os.path.join(Train_path, name)
            image = data_generator.load_image(path, INPUT_SHAPE)
            score_predict = self.model.predict(image[np.newaxis])[0]
            MM[cc,:] = score_predict
            cc += 1
        self.Score = MM 

        if save_name is not None:
            np.savetxt(save_name,MM)

    def LoadScore(self,save_name):
        self.Score = np.loadtxt(save_name)

    def EstimateThred(self,lr=0.001, save_name = 'thredshold'):
        thred = np.zeros((self.Y.shape[1])) + 0.5
        for _ in range(1000):
            Diff = self.Score - thred[np.newaxis, :]
            Y_ = np.where(Diff>0,1,0)
            error = GetThreshold.f1_loss(self.Y,Y_)
            thred = thred + error*lr
            #print(thred)
            #用眼睛看到收斂為止
        np.save(save_name,thred)

    @staticmethod
    def f1(y_true, y_pred):
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    @staticmethod
    def f1_loss(y_true, y_pred):
        tp = np.sum(y_true*y_pred, axis=0)
        fp = np.sum((1-y_true)*y_pred, axis=0)
        fn = np.sum(y_true*(1-y_pred), axis=0)

        p = tp / (tp + fp + np.finfo(float).eps)
        r = tp / (tp + fn + np.finfo(float).eps)

        f1 = 2*p*r / (p+r+np.finfo(float).eps)
        f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
        directection = np.where(np.mean(y_true - y_pred,axis = 0) > 0,-1,1)
        error = (1 - f1) * directection
        return error



def main():
    PATH_to_TRAINCSV=sys.argv[2]
    PATH_to_TRAIN=sys.argv[1]
    
    GS = GetThreshold(sys.argv[3])
    GS.GetY(PATH_to_TRAINCSV)
    GS.GetScore(Train_path= PATH_to_TRAIN,
        save_name = "Train_score")
    GS.LoadScore("Train_score")
    GS.EstimateThred(lr = 0.001,save_name = 'thredshold')

if __name__ == "__main__":
    main()
