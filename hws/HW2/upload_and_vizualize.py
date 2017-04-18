import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def read(file_name):

    df = pd.DataFrame()

    pattern = r'[(?!=.)]([a-z]*)'
    file_type = re.findall(pattern, file_name)[0]

    assert file_type == 'csv'
    with open(file_name, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for index,row in enumerate(reader):
            df = pd.concat([df,pd.DataFrame(row).transpose()])
   
    return df.reset_index(drop=True)

def read_file(file_name):

    pattern = r'[(?!=.)]([a-z]*)'
    file_type = re.findall(pattern, file_name)[0]
    
    if file_type == 'csv':
        df = pd.read_csv(file_name)
      
    elif file_type == 'xls':
        df = pd.read_excel(file_name)
    
    return df

def list_describe(df,optional_string=None):
    opt_columns = []
    all_cols = {}
    all_cols_caps = {}
    for index,column in enumerate(df.columns):
        if optional_string:
            if column.startswith(optional_string):
                opt_columns.append(column)

        print(df[str(column)].describe().to_frame(),'\n')
        if not index:
            continue
        all_cols[index] = camel_to_snake(column)
        all_cols_caps[index] = column

    return all_cols, all_cols_caps, opt_columns


def des_num_dep (data):
    '''
    Creates data frame with cumsum and percentage of 
    dependents by category
    Input: pandas data frame object
    Returns: new df with descriptive stats
    '''
    data = data['NumberOfDependents'].value_counts().to_frame()
    data['cumsum'] = data['NumberOfDependents'].cumsum()
    total = data['NumberOfDependents'].sum()
    data['percentage'] = (data['cumsum'] / total)*100 

    return data


def bin_feature(data, feature, given_range):
    '''
    Assumes a that a pandas dataframe is passed.
    Takes:
    data, a pandas dataframe that should be discretized, this procedure 
    will throw observations into "bins"
    feature
    num_bins
    '''

    bins = pd.DataFrame()
    discrete_feature = 'discrete_'+feature
    bins[discrete_feature] = pd.cut(data[feature], bins=given_range)

    df = pd.concat([data, bins], axis = 1)

    return df,discrete_feature

def create_plots(df,c_name,choice):
    
    possible_colors = ['r','b','g','orange','purple']
    choice = choice%len(possible_colors)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    df[c_name].plot(kind = "hist", alpha = 0.2, bins = 20, color = possible_colors[choice], ax = ax1); 
    ax1.set_title(c_name);
    ax1.grid()
    df.boxplot(column=c_name, ax = ax2); ax2.set_title('Boxplot of '+c_name)

    plt.tight_layout()

def create_hist_box(df,cols_dict,ignore):
    
    for index,col in enumerate(cols_dict.values()):
        if col in ignore:
            continue
        create_plots(df,col,index)

def create_line_graphs(df,feature,bins):
    df,discrete_feature = bin_feature(df, feature, bins)
    df[[discrete_feature,'SeriousDlqin2yrs']].groupby(discrete_feature).mean().plot(rot=70)
