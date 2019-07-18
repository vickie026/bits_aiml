import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as met
import math

dataset = pd.read_csv('concrete_data.csv')
#Deleting duplicate records
dataset.drop_duplicates('cement', inplace=True)
features = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'age', 'superplasticizer', 'fine_aggregate', 'coarse_aggregate']
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 8]
regressor = LinearRegression(normalize = True)
current_features = []
# Spliting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
best_features = []
size = len(X_train)
i = 0;
max = 0
prev_rmse= 0.0
m_mse = 0.0
while len(features) > 0 and i < len(features):
    #Fetching the feature
    current_features.append(features[i])
    X_in_test = X_train[current_features]
    y_in_test = y_train.values
    regressor.fit(X_in_test, y_in_test)
    scr = regressor.score(X_in_test, y_in_test)
    mse = met.mean_squared_error(y_in_test, regressor.predict(X_in_test))
    print('\n ADDED FEATURES ' + str(features[i]) + ' RMSE ', math.sqrt(mse * size))
    print('\n R2 SCORE IS ',scr)
    if scr > max or m_mse < prev_rmse :
        max = scr
        best_features.append(features[i])
        m_mse = mse
    features.remove(features[i])
    prev_rmse = mse
print(best_features)
print(' MAX ',max)
