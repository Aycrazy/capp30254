import csv
import pandas as pd
import re

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

def p_read_csv(file_name):
    df = pd.read_csv(file_name)
    return df