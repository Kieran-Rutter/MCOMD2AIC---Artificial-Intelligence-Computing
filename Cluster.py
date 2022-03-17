import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('Data-Cluster.csv')

plt.scatter(data['age'], data['sex'])
plt.xlim(0,150)
plt.ylim(0,1)

x = data.iloc[:,1:3] # 1t for rows and second for columns

kmeans = KMeans(3)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(data_with_clusters['age'],data_with_clusters['sex'],c=data_with_clusters['Clusters'],cmap='rainbow')

plt.show()