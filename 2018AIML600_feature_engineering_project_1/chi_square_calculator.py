import pandas as pd
import numpy as np
from scipy.stats import chi2
import csv
import math
import os

dataset1 = pd.read_csv('discretized_glass_features.csv')
dataset2 = pd.read_csv('glass_target.csv')
ref_values = [1,2,3,5,6,7]
headers = list(dataset1)
del headers[0]
# print(headers)
with open("frequencies.csv","w") as f :
    for m in headers :
        values = dataset1[m].sort_values().unique()
        for value in values :
            for i in ref_values :
                counter = 0
                for j in range(len(dataset1[m])) :
                    if dataset1[m][j] == value and dataset2['class'][j] == i :
                        counter += 1
                f.write(m+','+str(value)+','+str(i)+','+str(counter)+'\n')
chi_scores = {}
#reading frequencies.csv data and calculating chi square.
for feature in headers :
    columns = []
    rows = []
    uniq_val = dataset1[feature].sort_values().unique()
    size = len(ref_values)
    i = 0
    with open('frequencies.csv', 'rt')as f:
        data = csv.reader(f)
        for row in data :
            if row[0] == feature and i < (len(uniq_val)*size) :
                columns.append(int(row[3]))
    cols = np.asarray(columns)
    columns = cols.reshape(len(uniq_val), size)
#    print(columns)
    row_sums = []
    col_sums = []
    for i in range(len(columns)) :
        sum = 0
        for j in range(len(columns[i])) :
            sum += columns[i][j]
        col_sums.append(sum)
#    print(col_sums)
    for j in range(len(columns[0])) :
        sum = 0;
        for i in range(len(columns)) :
            sum += columns[i][j]
        row_sums.append(sum)
#    print(row_sums)
    grand_total = 0
    for i in col_sums :
        grand_total += i
#    print(grand_total)
    row_grand_total = 0
    for i in row_sums :
        row_grand_total += i
    if(grand_total != row_grand_total):
        print(' Something wrong happened with program...')
    #Calculating chi value
    chi_square = 0
    for i in range(len(columns)) :
        for j in range(len(columns[i])) :
            expected = (row_sums[j] * col_sums[i]) / grand_total
            chi_square += ((columns[i][j] - expected)**2)/expected
    chi = math.sqrt(chi_square)
    df = (len(columns) - 1) * (len(columns[0]) - 1)
    if feature == 'refractive_index' :
        feature = 'ri'
    print('Feature :: '+feature.upper()+'\tCHI :: '+str(round(chi_square, 2))+'\tdf :: ',df)
    p_val = 1 - chi2.cdf(round(chi_square, 2), df)
    chi_scores[feature] = p_val
os.remove('frequencies.csv')
sorted_p_list = sorted(chi_scores.items(), key=lambda x : (x[1], x[0]))
for x in sorted_p_list :
    print(str(x[0]).upper()+'\t'+str(x[1]))
print('completed')