import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plot
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('Data-Cluster.csv')

plot.scatter(data['capital-gain'], data['age'])
plot.xlim(0, 150000)
plot.ylim(0, 100)

x = data.iloc[:,1:3] # 1t for rows and second for columns (Slices)

kmeans = KMeans(30)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters
plot.scatter(data_with_clusters['capital-gain'],data_with_clusters['age'],c=data_with_clusters['Clusters'],cmap='rainbow')

plot.show()

#wcss=[]
#for i in range(1,15):
#    kmeans = KMeans(i)
#    kmeans.fit(x)
#    wcssiter = kmeans.inertia
#    wcss.append(wcss_iter)

#number_clusters = range(1,15)
#plot.plot(number_clusters,wcss)
#plot.title("The Elbow")
#plot.xlabel('Number of clusters')
#plot.ylabel('WCSS')

#plot.show()