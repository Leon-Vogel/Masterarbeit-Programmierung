import pickle
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import IsriDataset
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np

GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl"

isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))

df = pd.DataFrame()
for i in range(len(isri_dataset.data['Jobdata'])):
    df_temp = pd.DataFrame.from_dict(isri_dataset.data['Jobdata'][i], orient='index')
    df = pd.concat([df, df_temp])

df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10','time11','time12']] = pd.DataFrame(df.times.tolist(), index= df.index)
#print(df)
# 'time11',

df_cluster = df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10','time12']] #due_date
#df_cluster.to_excel('output.xlsx', index=False)
#pd.plotting.scatter_matrix(df_cluster, alpha=0.2, figsize=(15,15))
#plt.show()
kmeans = KMeans(n_clusters=8).fit(df_cluster)
centroids = kmeans.cluster_centers_
print(centroids)

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_cluster['time5'], df_cluster['time8'], df_cluster['time12'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
#ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

pd.plotting.scatter_matrix(df_cluster, alpha=0.2, c=kmeans.labels_.astype(float), figsize=(15,15))
plt.show()





'''plt.scatter(df_cluster['time1'], df_cluster['time5'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()'''

'''
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 18)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(df_cluster)
    kmeanModel.fit(df_cluster)
 
    distortions.append(sum(np.min(cdist(df_cluster, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / df_cluster.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(df_cluster, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / df_cluster.shape[0]
    mapping2[k] = kmeanModel.inertia_


for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


for key, val in mapping2.items():
    print(f'{key} : {val}')

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()
'''
