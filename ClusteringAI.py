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
X1 = pd.read_csv('Data-Cluster.csv')
X1.head()
X2 = pd.read_csv('Data-Cluster.csv')
X2.head()
X3 = pd.read_csv('Data-Cluster.csv')
X3.head()

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
X = X.filter(["fnlwgt", "amount" ], axis = 1)

model = KMeans(n_clusters= 4)
model.fit(X)


#plot the cluster
sns.scatterplot(data = X, x="fnlwgt", y="amount", c= model.labels_, cmap= 'rainbow', ax=ax2)
#Plots cluster mid points
sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'], ax=ax2)
ax2.set_xlim(0, 999999)
ax2.set_ylim(-0.1, 1.1)

#Assigns collumns
X1 = X1.filter(["workclass", "hours-per-week" ], axis = 1)

model = KMeans(n_clusters= 4)
model.fit(X1)

#plot the cluster
sns.scatterplot(data = X1, x="workclass", y= "hours-per-week", c= model.labels_, cmap= 'rainbow', ax=ax3)
#Plots cluster mid points
sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'], ax=ax3)
ax3.set_xlim(-1, 8)
ax3.set_ylim(0, 100)

#Assigns collumns
X2= X2.filter(["workclass", "fnlwgt" ], axis = 1)

model = KMeans(n_clusters= 4)
model.fit(X2)

#plot the cluster
sns.scatterplot(data = X2, x="workclass", y= "fnlwgt", c= model.labels_, cmap= 'rainbow', ax=ax4)
#Plots cluster mid points
sns.scatterplot(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'], ax=ax4)
ax4.set_xlim(-1, 8)
ax4.set_ylim(0, 999999)

plt.show()