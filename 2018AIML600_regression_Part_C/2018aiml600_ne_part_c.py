import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as met

dataset = pd.read_csv('concrete_data.csv')
#Deleting duplicate records
dataset.drop_duplicates('cement', inplace=True)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 8]
regressor = LinearRegression(normalize = True)
current_features = []
# Spliting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
regressor.fit(X_train, y_train)

y_predict = regressor.predict(X_test)

print(' COEFFICIENTS :: ', regressor.coef_)
print(' INTERCEPT :: ', regressor.intercept_)

score = met.r2_score(y_test, y_predict)
print(' SCORE :: ', score)

