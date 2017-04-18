import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from upload_and_vizualize import camel_to_snake
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def process_categorical(X,cat_col_list):
    cat_col_num_list = [loc for loc,c_name in enumerate(list(X.columns)) if c_name in cat_col_list]
    X = np.array(X)
    labelencoder_X = LabelEncoder()
    for cat_col in cat_col_num_list:
        X[:,cat_col] = labelencoder_X.fit_transform(X[:,cat_col])
    return X

def preprocess_by_mean(file_name):
    df = df.fillna(df.mean())
    return df


from sklearn.preprocessing import Imputer
def preprocess_imputer(credit_data):
    imputer = Imputer(missing_values='NaN', strategy = 'median', axis = 0)
    imputer = imputer.fit(credit_data)
    credit_data= imputer.transform(credit_data)
    return pd.DataFrame(credit_data).reset_index(drop=True)

def cross_vectors(df, var1, var2):
    return pd.crosstab(df[var1], df[var2])

def cross_validate(X,y):
    #Output:X_train, X_test, y_train, y_test 
    return train_test_split(X, y, test_size = .2, random_state = 0)

def model_logistic(X_train, y_train, X_test):

    '''
    With training and testing data and the data's features and label, select the best
    features with recursive feature elimination method, then
    fit a logistic regression model and return predicted values on the test data
    and a list of the best features used.
    '''
    
    model = LogisticRegression()
    rfe = RFE(model)
    rfe = rfe.fit(X_train, y_train)
    predicted = rfe.predict(X_test)
    best_features = rfe.get_support(indices=True)
    return predicted, best_features

from sklearn.metrics import accuracy_score 

#currently borrowed from Hector
def accuracy(observed, predicted):
    '''
    Takes:
    predicted, a list with floats or integers with predicted values
    observed, a list with floats or integers with observed values 
    Calculates the accuracy of the predicted values.
    '''
    return accuracy_score(observed, predicted)

