# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

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

def solve2_3():
    print("the number of sample: ", train_data.shape[0])
    print("the number of features: ", train_data.shape[1])
    print("PassengerId 7' ", train_data['Name'][6])

def solve2_5(data):
    # drop missing columns
    print(data.isnull().sum())
    preprocessed_data = data.drop(["Age", "Cabin"], axis=1)
    print(preprocessed_data.isnull().sum())
    
    # fill 
    print(preprocessed_data["Embarked"].value_counts())
    print(preprocessed_data["Embarked"].mode())
    
    preprocessed_data["Embarked"].fillna(preprocessed_data["Embarked"].mode()[0], inplace=True)
    print(preprocessed_data.isnull().sum())
    
def main():
    train_data = read_train_data()
    refined_data = refine_data(train_data)
    
    X = refined_data.values[:, 1:]
    y = refined_data.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    classifier = MLPClassifier()
    classifier.fit(X_train_transformed, y_train)
    print("Accuracy: ", classifier.score(X_test_transformed, y_test))

    """

    dtc = DecisionTreeClassifier()
    svc = SVC()
    
    dtc.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    
    cv_score_dtc = cross_val_score(dtc, X, y, cv=10)
    cv_score_svc = cross_val_score(svc, X, y, cv=10)
    
    print("------------------------------------Not Standardized------------------------------------")
    print("Decition Tree Accuracy: ", cv_score_dtc.mean())
    print("SVM Accuracy: ", cv_score_svc.mean())
    """

    """
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    
    trained_dtc = DecisionTreeClassifier()
    trained_svc = SVC()
    
    trained_dtc.fit(X_train_transformed, y_train)
    trained_svc.fit(X_train_transformed, y_train)

    cv_score_standardized_dtc = cross_val_score(trained_dtc, X, y, cv=10)
    cv_score_standardized_svc = cross_val_score(trained_svc, X, y, cv=10)
    print("------------------------------------Standardized------------------------------------")
    print("Decition Tree Accuracy: ", cv_score_standardized_dtc)
    print("SVM Accuracy: ", cv_score_standardized_svc)
    
    #print(train_data.describe())
    #print(train_data.head(10))
    """
    
    """
    pipe_svc = Pipeline([('scaler', StandardScaler()),
                         ('classifier', SVC(random_state=0))])
    
    param_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    parameters = {'classifier__C': param_grid,
                  'classifier__gamma': param_grid}
    
    gs = GridSearchCV(pipe_svc,
                      parameters,
                      cv=10)
    
    gs = gs.fit(X_train, y_train)
    
    print(gs.best_params_)
    
    cls = gs.best_estimator_
    cls.fit(X_train, y_train)
    print("Grid Searched SVM Accuracy: ", cls.score(X_test, y_test))
    """
    
    

if __name__ == '__main__':
    main()