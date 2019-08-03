from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

X = pd.read_csv("glass_features.csv")
y = pd.read_csv("glass_target.csv")

print("Actual Values\n",X.iloc[0,:])

# Create the sbs object and select best 4 features
feature_size = 5
knn = KNeighborsClassifier(n_neighbors=feature_size)
# the param forward when set to False will do sequential backward selection
sbs = SFS(knn,
           k_features=feature_size,
           forward=False,
           floating=False,            
           scoring='accuracy',
           )
try :
    sbs = sbs.fit(X, y)
except :
    print(' Error in fit')

print("Best " + str(feature_size) + " features: ", sbs.k_feature_idx_)
print("Feature names : ", sbs.subsets_[feature_size]['feature_names'])
print("Feature cv scores : ", sbs.subsets_[feature_size]['cv_scores'])
print("K_score ", sbs.k_score_)