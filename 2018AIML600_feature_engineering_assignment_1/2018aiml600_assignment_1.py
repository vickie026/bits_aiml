#importing libraries
import csv
import pandas as pd
import numpy as np
import re

#It is assumed that once a column is dropped due to missing values or None values >= 0.4, it is not processed any further.

with open("final_dataset.csv") as f:
    reader = csv.reader(f, delimiter=",")
    with open("modified_final_dataset.csv", "w", newline="") as fo:
        writer = csv.writer(fo)
        for rec in reader:
            writer.writerow(map(str.strip, rec))

# to display the total number columns present in the dataset
pd.set_option('display.max_columns', None)

# let's load the final dataset
dataset = pd.read_csv('modified_final_dataset.csv')
column_names = list(dataset)
#Considering None as missing value
dataset[column_names] = dataset[column_names].replace("None", np.NaN)
#getting the missing data percentage for each column
total_missing_data_percentage = dataset.isnull().mean()
#droping data from dataset where total missing data percentage >= 40%
processing_columns = [(i >= 0.4) for i in total_missing_data_percentage]
deleting_columns = []
#Getting all columns which needs to be dropped as list.
for i in range(len(processing_columns)) :
    if processing_columns[i] :
        deleting_columns.append(column_names[i])
#dropping the columns from dataset
dataset.drop(deleting_columns, axis = 1, inplace=True)

pattern = re.compile('\W+ ')

#Converting property_price into lacs and removing rupee signs
#for i in range(len(dataset['property_price'])) :
#    dataset['property_price'][i] = re.sub(pattern, '', dataset['property_price'][i])
dataset['property_price'] = dataset['property_price'].str.replace(pattern,"")
dataset['property_price'] = dataset['property_price'].str.replace(" Lac","*1")
dataset['property_price'] = dataset['property_price'].str.replace(" Cr", "*100")

new_property_price = dataset['property_price'].str.split("*", n=1, expand=True)
dataset['temp_property_price1'] = pd.to_numeric(new_property_price[0])
dataset['temp_property_price2'] = pd.to_numeric(new_property_price[1])

dataset.eval('property_price = temp_property_price1 * temp_property_price2', inplace=True)
dataset.drop(['temp_property_price1', 'temp_property_price2'], axis=1, inplace=True)
# Filling empty values with 0
dataset = dataset.fillna(value="0 ")

#Label Encoding to label categorical data with 0,1,...class_n-1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['transaction'] = le.fit_transform(dataset['transaction'])
dataset['furnishing'] = le.fit_transform(dataset['furnishing'])

#Processing floor variable
dataset['floor'] = dataset['floor'].str.replace("Ground", "0")
dataset['floor'] = dataset['floor'].str.replace(" out of ", "#")
dataset['floor'] = dataset['floor'].str.replace(" floors", "")
#for i in range(len(dataset['floor'])) :
#    dataset['floor'][i] = round(eval(dataset['floor'][i]),2)
new_floor = dataset['floor'].str.split("#", n = 1, expand = True)
dataset['floor1'] = pd.to_numeric(new_floor[0])
dataset['floor2'] = pd.to_numeric(new_floor[1])
dataset = dataset.fillna(value=1)
dataset.eval('floor = floor1 / floor2', inplace = True)
dataset.drop(['floor1', 'floor2'], axis = 1, inplace = True)
dataset = dataset.round({'floor':2})
dataset.to_csv(r'2018aiml600_final_dataset.csv', index=None, header=True)