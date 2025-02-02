
import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl
from upload_and_vizualize import camel_to_snake


def get_desired_features(dict, desired_features):
    '''
    Given a dictionary with the keys as the number column of the feature
    and the feature title, creates a list of the features

    Input:
        dict = dictionary of column value
        desired_features = list of desired features keys
    Output:
        feature keys_list
    '''

    features = [key for key,value in dict.items() if value in desired_features]
    return features

def check_na(df):
    '''
    Given a dataframe, function outputs a dataframe identifying the number of
    non-NA and NA values
    '''

    df_lng = pd.melt(df)
    null_vars = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_vars)


def check_diff(df, column1, column2):
    '''
    Given a dataframe and two column names, function optuts the number of differences between a particular
    column of interest, and its intersection with another column.
    column1 = column of interest
    '''
    null2 = df[df[column1].isnull()| df[column2].isnull()]
    null1 = df[df[column1].isnull()]
    diff = set(null1.index)-set(null2.index)
    return(len(diff))

def describe_extremes(df,column, value):
    '''
    Given a dataframe, a column of interest, and a relative extreme value,
    this function will output the ratio of such extreme values to the rest
    of the dataframe rows
    '''

    column_str = str(column)
    very_high = df[column_str] > value
    print(df[column_str].debt_ratio.describe())
    print(len(df[column_str])/len(df))

def add_categoricals(dict, new_categories):
    '''
    Add the name of new columns for categorical variables to the dictionary
    of column names
    '''

    count=1
    for cat in new_categories:
        dict[len(dict)+count] = cat
'''
def update_keys(dict):
    return {key-1:value for key,value in dict.items()}
'''

def add_dummies(df,need_dummies):
    '''
    Given a dataframe, this function makes dummy variables out of given categorical
    variables
    Input:
        df = pandas dataframe
        need_dummies = columns that need dummy variable columns
    '''
    
    new_cols = pd.DataFrame()
    for col in need_dummies:
        #print(np.array(df[col]))
        new_cols = pd.concat([new_cols.reset_index(drop=True),pd.get_dummies(df[col])],axis=1)

    df_w_dummies = pd.concat([df,new_cols],axis=1).reset_index(drop=True)
    return df_w_dummies