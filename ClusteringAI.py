#imports libary
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns

#sets the style of the grid
sns.set_style("darkgrid")

#Loads the dataset
X = pd.read_csv('Data-Cluster.csv')
X.head()

#Runs an elbow to decide the correct amount of clusters
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#Assigns collumns
X = X.filter(["fnlwgt", "age" ], axis = 1)

model = KMeans(n_clusters= 5)
model.fit(X)

#plot the cluster
sns.scatterplot(data = X, x="fnlwgt", y= "age", c= model.labels_, cmap= 'rainbow' )
#Plots cluster mid points
sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'])
plt.show()