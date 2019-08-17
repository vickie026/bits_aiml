from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def draw_and_save_figure(pca, figurename) :
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig(figurename+'.png')

def myplot(score, m, p):
    xs = score[:, m]
    ys = score[:, p]
    plt.scatter(xs , ys, c=y)  # without scaling
    plt.xlabel("PC{}".format(m + 1))
    plt.ylabel("PC{}".format(p + 1))
    plt.grid()

def myplot3d(pca):
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.figure(figsize=(16, 12)).gca(projection='3d')
    ax.scatter(
        xs=pca[:, 0],
        ys=pca[:, 1],
        zs=pca[:, 2],
        c=y,
        cmap='tab10'
    )
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))
    ax.set_zlabel("PC{}".format(3))
    plt.savefig("PC1_PC2_PC3.png")

dataset = load_breast_cancer()
X = dataset['data']
y = dataset['target']
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X)

draw_and_save_figure(pca, "explained_var_components")
print(" explained variance :: ", np.round(pca.explained_variance_, decimals=3) * 100)
print(" explained variance ratio :: ", np.round(pca.explained_variance_ratio_, decimals=3) * 100)
print(" Cumulative sum of variance :: ",np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100))

# PC1_PC2. 
myplot(X_train_pca[:, 0:2], m = 0, p = 1) 
plt.savefig("PC1_PC2.png")

# PC1_PC3. 
myplot(X_train_pca[:, 0:3], m=0, p=2) 
plt.savefig("PC1_PC3.png")

# PC2_PC3. 
myplot(X_train_pca[:, 0:3], m=1, p=2) 
plt.savefig("PC2_PC3.png")

#PC1_PC2_PC3_3D
myplot3d(X_train_pca[:, 0:3])