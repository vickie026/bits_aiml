from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from  sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

X = pd.read_csv("glass_features.csv")
y = pd.read_csv("glass_target.csv")

print("Actual Values\n", X.iloc[0, :])
svc = LinearSVC()
feature_size = 6
selector = RFE(svc,feature_size,1)
selector = selector.fit(X, y)
features_selected_idx = []
for i in range(len(selector.support_)) :
    if selector.support_[i] :
        features_selected_idx.append(i)
print(" Selected features Index ", features_selected_idx)
selected_features = [list(X)[i] for i in features_selected_idx]
print(" Selected features " , selected_features)
y_pred = selector.predict(X)
rmse = sqrt(mean_squared_error(y, y_pred))
print(rmse)