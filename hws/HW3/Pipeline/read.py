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
        for row in reader:
            df = pd.concat([df,pd.DataFrame(row)])
        df = pd.DataFrame(reader)
    #df = csv.read_csv(file_name)

    return df

def p_sread_csv(file_name):
    df = pd.read_csv(file_name)
    return df