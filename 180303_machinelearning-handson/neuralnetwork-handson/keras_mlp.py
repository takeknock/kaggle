# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

from keras.callbacks import ModelCheckpoint

from data_loader import get_x_train_transformed, get_x_test_transformed, get_y_train, get_y_test

def create_model():
    model = Sequential()
    model.add(Dense(15, activation='relu', input_dim=7))
    model.add(Dense(15, activation='relu', input_dim=15))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    return model

def train_and_save(model):
    tb_cb = TensorBoard(log_dir='tb_log4/')
    
    fpath = './model/weights.{epoch:3d}-{loss:2f}-{val_loss:2f}.hdf5'    
    cp_cb = ModelCheckpoint(filepath=fpath, period=5)
 
    
    model.fit(X_train_transformed, 
              y_train,
              epochs=1000,
              batch_size=64,
              validation_split=0.2      ,
              callbacks=[tb_cb, cp_cb])
    
    return model
   
def load_model():
    model = keras.models.load_model('./model/weights.120-0.442883-0.452981.hdf5')
    return model

def main():
    X_train_transformed = get_x_train_transformed()
    X_test_transformed = get_x_test_transformed()
    y_train = get_y_train()
    y_test = get_y_test()
    
    #model = create_model()
    #model = train_and_save(model)
    
    model = load_model()
    
    score = model.evaluate(X_test_transformed, y_test)
    print('loss: ', score[0])
    print('accuracy: ', score[1])
    
    print(model.predict(X_test_transformed[0:10, :]))
    
    
if __name__ == '__main__':
    main()