from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as met
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv('father_son_heights.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

#Spliting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
print('\n Mini Batch GD Regressor')


def calc_rmse(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())


def calc_rse(targets, predictions):
    df = len(targets) - 2
    return math.sqrt(np.sum((predictions - targets) ** 2)/df)

size = 55
#Batch creation
def create_batch(X, y, size) :
       indx = np.random.choice(len(X), len(y), replace = False)
       x_points = X[indx]
       y_points = y[indx]
       [(x_points[i:i+size,:], y_points[i:i+size,] ) for i in range(0, len(X), size)]
       yield x_points
       yield y_points

batch_num = int(len(y)/size)
n_itr = 1000

x_batch, y_batch = create_batch(X, y, size)

#Creating SGD
mb_sgd =SGDRegressor(eta0=0.001, random_state=15, tol = 1e-2)
for ep in range(n_itr) :
       batch_indexer = 0
       for i in range(1, batch_num + 1) :
              mb_sgd.partial_fit(
                  x_batch[batch_indexer:i*size].reshape(-1,1)/min(X), y_batch[batch_indexer:i*size]/min(y))
              batch_indexer += size
#Getting the coefficients and intercepts
pre_coef = mb_sgd.coef_
pre_inter = mb_sgd.intercept_*min(y)

print(' COEF :: ', pre_coef)
print(' INTERCEPT :: ',pre_inter)

y_pred = mb_sgd.predict(X/min(X))
y_regressor = pre_inter + pre_coef*X


r2 = met.r2_score(y/min(y), y_pred)
rmse = calc_rmse(y, y_regressor)

print(' R2 Score :: ', r2)
print(' RMSE :: ',rmse)
factor = math.sqrt(len(y)/(len(y) - 2))
print(' RSE :: ', (rmse*factor))

# PLOTTING THE POINTS
plt.scatter(X, y, color='red', marker='.')
plt.plot(X, y_regressor, color='blue')
plt.xlabel("Father's Height")
plt.ylabel("Son's Height")
plt.title('Father-Son Heights Figure')
plt.savefig("minibatch_line.png")