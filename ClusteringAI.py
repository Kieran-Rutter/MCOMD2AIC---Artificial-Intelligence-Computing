#imports libary
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns

#sets the style of the grid
sns.set_style("darkgrid")

#Sets the subplots
fig, ([ax1, ax2],[ax3, ax4]) = plt.subplots(2,2, figsize=(16,6))

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
ax1.plot(K, distortions, 'bx-')
ax1.set_xlabel('k')
ax1.set_ylabel('Distortion')
ax1.set_title('The Elbow Method showing the optimal k')


#Assigns collumns
X = X.filter(["fnlwgt", "age" ], axis = 1)

model = KMeans(n_clusters= 5)
model.fit(X)

#plot the cluster
sns.scatterplot(data = X, x="fnlwgt", y= "age", c= model.labels_, cmap= 'rainbow', ax=ax2)
#Plots cluster mid points
sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'], ax=ax2)

#plot the cluster
sns.scatterplot(data = X, x="fnlwgt", y= "age", c= model.labels_, cmap= 'rainbow', ax=ax3)
#Plots cluster mid points
sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'], ax=ax3)

#plot the cluster
sns.scatterplot(data = X, x="fnlwgt", y= "age", c= model.labels_, cmap= 'rainbow', ax=ax4)
#Plots cluster mid points
sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'], ax=ax4)

plt.show()