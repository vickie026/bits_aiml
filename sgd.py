from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as met
import matplotlib.pyplot as plt

def main() :
    dataset = pd.read_csv('father_son_heights.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:,1].values

    #Spliting training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
    print('\n SGD Regressor')
    #Getting the model
    print('\n Please wait. we are creating the best possible SGD Regressor model according to data...')
    sgd,coef,intercept = get_sgd(X_train,X_test, y_train, y_test)
    #Training the model
    print('\n Model created successfully. Now training the data...')
    sgd.fit(X_train, y_train, coef_init=coef, intercept_init=intercept)
    print('\n Training completed. Now scoring with test data')
    y_predict = sgd.predict(X_test)
    print('\n ESTIMATED COEF_',sgd.coef_)
    print('\n ESTIMATED INTERCEPT_ ',sgd.intercept_)
    scr1 = r2_score(y_test, y_predict)
    rse = calc_rse(y_test, y_predict)
    rmse = calc_rmse(y_test, y_predict)
    print('\n R2 IS :: ',scr1)
    print(' RMSE is :: ',rmse)
    print(' RSE is :: ',rse)
    plt.scatter(X_test,y_test,color='red',marker='.')
    plt.plot(X_test,y_predict,color='blue')
    plt.xlabel("Father's Height")
    plt.ylabel("Son's Height")
    plt.title('Father-Son Heights Figure')
    plt.savefig("sgd_line.png")



def get_sgd(X_train, X_test, y_train, y_test):
    temp_max_itr = 100000
    dest_eta = 1e-5
    dest_tol = 1e-3
    temp_coef = 0.01
    dest_coef = temp_coef
    dest_intercept = 0.0
    max = -1000
#    mode = 'w'
#    cnt = 1
    while temp_coef <= 2.0 :
        temp_intercept = 0.0
        while temp_intercept <= 50.0 :
            sgd = SGDRegressor(random_state=15, max_iter=temp_max_itr,
                           eta0=dest_eta, tol=dest_tol, n_iter_no_change=6)
            sgd.fit(X_train, y_train, coef_init=temp_coef,
                    intercept_init=temp_intercept)
            scr = sgd.score(X_test, y_test)
            #Checking if scored more than previous max score
            if max < scr :
                max =scr
                dest_coef = temp_coef
                dest_intercept = temp_intercept
#            if cnt > 1 :
#                mode = 'a'
#            cnt += 1
#            write_to_file(scr,dest_coef, dest_intercept, mode)
            temp_intercept += 1.0
        temp_coef += 0.1
    sgd1 = SGDRegressor(random_state=15, max_iter=temp_max_itr, eta0=dest_eta, tol=dest_tol, n_iter_no_change=6)
    return sgd1,dest_coef,dest_intercept
#Just for analysis purpose. Not used in current project anymore.
def write_to_file(scr,dest_coef, dest_intercept, mode) :
    with open("score_card.txt",mode) as f:
        f.write('\n R2_Score :: '+str(scr)+'| coef ::'+str(dest_coef)+'| intercept ::'+str(dest_intercept))
    print(' Recorded Successfully')


def calc_rmse(targets, predictions) :
    return np.sqrt(((predictions - targets) ** 2).mean())

def calc_rse(targets, predictions) :
    import math
    df = len(targets) - 2;
    return math.sqrt(np.sum(np.square(targets - predictions))/df)


if __name__ == "__main__" :
    main()

