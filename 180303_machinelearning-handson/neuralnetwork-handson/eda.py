# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 01:59:00 2018

@author: train2
"""

from data_loader import *
import matplotlib
import matplotlib.pyplot as plt


def main():
    train_data = get_train_data()
    
    #print(train_data.head())
    X_train_transformed = get_x_train_transformed()
    X_test_transormed = get_x_test_transformed()
    y_train = get_y_train()
    y_test = get_y_test()
    
    # Extract survived 1
    surviver_df = train_data[train_data.Survived==1]
    train_data.set_index("PassengerId")
    
    male_data = train_data[train_data.Sex=="male"]
    print(male_data.head())
    train_data.plot.hist(bins=20, figsize=(18, 16), color="#f1b7b0")
    
if __name__ == '__main__':
    main()