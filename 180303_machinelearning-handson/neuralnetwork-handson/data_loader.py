# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 07:54:41 2018

@author: train2
"""

import pandas as pd
import numpy as np

def get_train_data():
    return pd.read_csv('C:/Users/train2/train.csv')

def get_x_train_transformed():
    data = np.loadtxt('C:/Users/train2/nn-handson/data/X_train_transformed.csv', delimiter=',')
    return data

def get_x_test_transformed():
    data = np.loadtxt('C:/Users/train2/nn-handson/data/X_test_transformed.csv', delimiter=',')
    return data

def get_y_train():
    data = np.loadtxt('C:/Users/train2/nn-handson/data/y_train.csv', delimiter=',')
    return data

def get_y_test():
    data = np.loadtxt('C:/Users/train2/nn-handson/data/y_test.csv', delimiter=',')
    return data