# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 01:13:32 2018

@author: train2
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def read_train_data():
    return pd.read_csv('C:/Users/train2/train.csv')

def refine_data(data):
    data = refine_nominaldata(data)
    data = refine_categoricaldata(data)
    return data

def refine_nominaldata(data):
    data = data.drop(["Age", "Cabin"], axis=1)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    return data
    
def refine_categoricaldata(data):
    data = data.drop(["Name", "Ticket", "PassengerId"], axis=1)
    one_hot = pd.get_dummies(data["Sex"])
    data = pd.concat((data, one_hot), axis=1)
    data = data.drop(["Sex", "male"], axis=1)
    one_hot = pd.get_dummies(data["Embarked"])
    data = pd.concat((data, one_hot), axis=1)
    data = data.drop(["Embarked", "S"], axis=1)
    return data

def standardize(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    return X_train_transformed, X_test_transformed

def save(X_train, X_test, y_train, y_test):
    np.savetxt('C:/Users/train2/nn-handson/data/X_train_transformed.csv', X_train, delimiter=',')
    np.savetxt('C:/Users/train2/nn-handson/data/X_test_transformed.csv', X_test, delimiter=',')
    np.savetxt('C:/Users/train2/nn-handson/data/y_train.csv', y_train, delimiter=',')
    np.savetxt('C:/Users/train2/nn-handson/data/y_test.csv', y_test, delimiter=',')

def main():
    train_data = read_train_data()
    refined_data = refine_data(train_data)
    
    X = refined_data.values[:, 1:]
    y = refined_data.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train_transformed, X_test_transformed = standardize(X_train, X_test)
    save(X_train_transformed, X_test_transformed, y_train, y_test)
    print('Text Saved!!!')
    

if __name__ == '__main__':
    main()
