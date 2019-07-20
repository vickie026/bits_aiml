import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.simplefilter("ignore")

# Defining a lasso polynomial regressor
def polynomial_lasso_regressor(deg, X_train, X_test, y_train, y_test) :
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X_train)
    X_test_poly = poly_reg.fit_transform(X_test)
    poly_reg.fit(X_poly, y_train)
    # creating Ridge regressor with default alpha
    from sklearn.linear_model import Ridge
    ridge = Ridge()
    ridge = Ridge()
    ridge.fit(X_poly, y_train)

    # Predicting training and testing values
    y_predicted = ridge.predict(X_poly)
    y_test_predicted = ridge.predict(X_test_poly)

    if deg == 10 :
        from sklearn.linear_model import LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y_train)
        linear_train_score = lin_reg.score(X_poly, y_train)
        linear_test_score = lin_reg.score(X_test_poly, y_test)
        ridge_train_score = ridge.score(X_poly, y_train)
        ridge_test_score = ridge.score(X_test_poly, y_test)
        print(' Poly linear_train_score ', linear_train_score)
        print(' Poly linear_test_score ', linear_test_score)
        print(' Poly ridge_train_score ', ridge_train_score)
        print(' Poly ridge_test_score ', ridge_test_score)
    from sklearn.metrics import mean_squared_error
    # Getting RMSE
    rmse_train = sqrt(mean_squared_error(y_train, y_predicted))
    rmse_test = sqrt(mean_squared_error(y_test, y_test_predicted))

    # returning training rmse and test rmse
    return rmse_train, rmse_test


def main() :

    # Getting training and testing datasets already provided
    train_dataset = pd.read_csv('project - part D - training data set.csv')
    test_dataset = pd.read_csv('project - part D - testing data set.csv')

    # Training set.
    X_train = train_dataset.iloc[:, 1:2].values
    y_train = train_dataset.iloc[:, 2].values

    # Testing set
    X_test = test_dataset.iloc[:, 1:2].values
    y_test = test_dataset.iloc[:, 2].values

    max_deg = 10
    rmse_train = []
    rmse_test = []
    for i in range(1, max_deg + 1) :
        rmse_tr, rmse_ts = polynomial_lasso_regressor(i, X_train, X_test, y_train, y_test)
        rmse_train.append(rmse_tr)
        rmse_test.append(rmse_ts)
    print(' RMSE Value for degree = 10 on training data', rmse_train[9])  
    print(' RMSE Value for degree = 10 on testing data', rmse_test[9])    
    plt.plot([i for i in range(1, max_deg + 1)], rmse_train, color="blue", label='train_data', marker='o', linestyle='solid')
    plt.plot([i for i in range(1, max_deg + 1)], rmse_test, color="red", label='test_data', marker='o', linestyle='solid')
    plt.legend(loc='upper left')
    plt.title(" DEGREES-VS-ERMS ")
    plt.xlabel(" DEGREES ")
    plt.ylabel(" ERMS ")
    plt.savefig('ridge_deg_rmse.png')
    plt.show()


if __name__ == "__main__" :
    main()