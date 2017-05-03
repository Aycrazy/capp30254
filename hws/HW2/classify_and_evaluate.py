import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import time

matplotlib.style.use('ggplot')
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from upload_and_vizualize import camel_to_snake
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn.metrics import *
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve

# if you're running this in a jupyter notebook, print out the graphs - wise words from R Ghani
NOTEBOOK = 0

def split_data_cust(df, use_seed=None):
    '''
    Given a pandas datframe sorted by column to have the first column be
    the outcome column, and the following columns to be potential features
    split the fame into training sets and testing sets
    '''
    if use_seed:
        np.random.seed(seed=20)
    test_set = np.random.uniform(0,1,len(df)) > .75
    X_train = df[test_set==False].ix[:,1:]
    X_test = df[test_set==True].ix[:,1:]
    y_train = df[test_set==False].ix[:,1]
    y_test = df[test_set==True].ix[:,1]
    return X_train,X_test,y_train,y_test

def process_categorical(df,cat_col_list):
    '''
    Given a pandas dataframe. This function will process continuous variables
    and transform them into categorical variables
    Inputs:
        df = pandas dataframe
        cat_col_list = list of columns you want categorical variables for
    '''

    cat_col_num_list = [loc for loc,c_name in enumerate(list(X.columns)) if c_name in cat_col_list]
    X = np.array(df)
    labelencoder_X = LabelEncoder()
    for cat_col in cat_col_num_list:
        X[:,cat_col] = labelencoder_X.fit_transform(X[:,cat_col])
    return X

def preprocess_by_mean(df):
    '''
    Given a pandas dataframe, this function will fillna with the mean
    '''
    df = df.fillna(df.mean())
    return df


def preprocess_imputer(df,by_strategy):
    '''
    Given a pandas dataframe and a imputation method,
    this will impute values for NaN values with values according the strategy
    (ie 'median')
    '''
    imputer = Imputer(missing_values='NaN', strategy = by_strategy, axis = 0)
    imputer = imputer.fit(df)
    df= imputer.transform(df)
    return pd.DataFrame(df).reset_index(drop=True)

def cross_vectors(df, var1, var2):
    '''
    Given a pandas dataframe and variable names
    create a crosstable dataframe
    '''
    return pd.crosstab(df[var1], df[var2])

def split_data(X,y,size):
    '''
    Given two dataframes ouptut a training and testing set given size
    '''

    #Output:X_train, X_test, y_train, y_test 
    return train_test_split(X, y, test_size = size, random_state = 0)


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

def define_clfs_params(grid_size):
    """
    Creates dictionaries classifiers and a grid used to inform the user of
    the strengths and weaknesses of each model
    Source: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    """


    clfs = {'LR': LogisticRegression(penalty='l1', C=1e5),
        'KNN': KNeighborsClassifier(n_neighbors=3), 
        'DT': DecisionTreeClassifier(),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10)}

    large_grid = { 
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]}}
    
    small_grid = { 
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

def generate_binary_at_k(y_scores, k):
    '''
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    '''
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def do_learning(X_training, Y_training, X_test, Y_test, reference_dic, model_class):

    '''
    credit: Juan Arroyo-Miranda & Dani Alcala

    With training and testing data select the best
    features with recursive feature elimination method, then
    fit a classifier and return a tuple containing the predicted values on the test data
    and a list of the best features used.
    '''
    
    model = model_class
    # Recursive Feature Elimination
    rfe = RFE(model)
    rfe = rfe.fit(X_training, Y_training)
    
    best_features = rfe.get_support(indices=True)

    best_features_names = [reference_dic[i] for i in best_features]

    predicted = rfe.predict(X_test)
    expected = Y_test

    accuracy = accuracy_score(expected, predicted)
    return (expected, predicted, best_features_names, accuracy)


def run_mods(models_to_run, clfs, grid, X_training, Y_training, X_test, Y_test, print_plots='no'):

    '''
    '''
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))

    for index, clf in enumerate([clfs[x] for x in models_to_run]):

        parameter_cols = grid[models_to_run[index]]

        for p in ParameterGrid(parameter_cols):
            try:
                clf.set_params(**p)

                start_time_training = time.time()
                clf.fit(X_training, Y_training)
                train_time = time.time() - start_time_training

                start_time_predicting = time.time()
                y_pred_prob = clf.predict_proba(X_test)[:,1]
                predict_time = time.time() - start_time_predicting

                roc_score = roc_auc_score(Y_test, y_pred_prob)
                y_pred_prob_sorted, y_test_sorted = zip(*sorted(zip(y_pred_prob, Y_test), reverse=True))

                results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(Y_test, y_pred_prob),
                                                       precision_at_k(y_test_sorted,y_pred_prob_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_prob_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_prob_sorted,20.0)]
                if print_plots == 'yes':
                    plot_precision_recall_n(Y_test,y_pred_prob,clf)
            except IndexError as e:
                print('Error:',e)
                continue

    return results_df


def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    '''

    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

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

def plot_confusion_matrix(data, col_name, labels, model_name):
    '''
    Given a pandas dataframe with a confusion confusion_matrix
    and a list of axis lables plot the results
    '''
    sn.set(font_scale=1.4)#for label size

    xticks =  labels
    yticks =  labels
    ax = plt.axes()
    sn.heatmap(data, annot=True,annot_kws={"size": 16}, linewidths=.5, xticklabels = xticks,  
              yticklabels = yticks, fmt = '')
    ax.set_title('Confusion Matrix for' + ' ' + model_name + col_name)

def create_confusion_matrix(df_y_test,df_y_pred, col_name, labels, model_name):
    '''
    Given an actual set of y values (based on the test set) and a predicted set of y values 
    (based on the test set), a column name, and a column name this function will produce
    a confusion matrix and then plot that matrix, utilizing the plot_confusion_matrix function.
    '''

    actual = pd.Series(df_y_test[col_name], name = 'Actual')
    predicted = pd.Series(df_y_pred[col_name], name='Predicted')
    array = confusion_matrix(actual, predicted)
    df_cxm = pd.DataFrame(array, range(2), range(2))
    plot_confusion_matrix(df_cxm,col_name, labels, model_name)

