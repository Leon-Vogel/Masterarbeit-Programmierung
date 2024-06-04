import pickle
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import IsriDataset
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os

class cluster_kmeans():
    def __init__(self, n_cluster, path="isri_optimizer/rl_sequential_agent/cluster_models/"):
        if os.path.exists(path+"kmeans_model_n"+str(n_cluster)+".pkl"):
            with open(path+"kmeans_model_n"+str(n_cluster)+".pkl", "rb") as f:
                self.kmeans = pickle.load(f)
        else:
            GA_SOLUTIONS_PATH = "isri_optimizer/rl_sequential_agent/IsriDataset.pkl"

            isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))

            df = pd.DataFrame()

            for i in range(len(isri_dataset.data['Jobdata'])):
                df_temp = pd.DataFrame.from_dict(isri_dataset.data['Jobdata'][i], orient='index')
                df = pd.concat([df, df_temp])

            df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10','time11','time12']] = pd.DataFrame(df.times.tolist(), index= df.index)
            df = df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10', 'time11', 'time12']]

            self.kmeans = KMeans(n_clusters=n_cluster).fit(df)
            with open(path+"kmeans_model_n"+str(n_cluster)+".pkl", "wb") as f:
                pickle.dump(self.kmeans, f)
        
    def label(self, data):
        data = np.array(data)
        data = data.reshape(1, -1)
        return self.kmeans.predict(data)



#Test:
'''Cluster = cluster_kmeans(8)

GA_SOLUTIONS_PATH = "isri_optimizer/rl_sequential_agent/IsriDataDict.pkl"
isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
df = pd.DataFrame()

for i in range(len(isri_dataset['Jobdata'])):
    df_temp = pd.DataFrame.from_dict(isri_dataset['Jobdata'][i], orient='index')
    df = pd.concat([df, df_temp])

df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10','time11','time12']] = pd.DataFrame(df.times.tolist(), index= df.index)
df = df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10', 'time11', 'time12']]
daten = df.sample(frac=0.001).to_numpy()
y = Cluster.label(daten)
print(y)'''

class cluster_neighbour():
    def __init__(self, n_cluster, path="isri_optimizer/rl_sequential_agent/cluster_models/"):
        if os.path.exists(path+"knn_model_n"+str(n_cluster)+".pkl"):
            with open(path+"knn_model_n"+str(n_cluster)+".pkl", "rb") as f:
                self.knn_model = pickle.load(f)
        else:
            GA_SOLUTIONS_PATH = "isri_optimizer/rl_sequential_agent/IsriDataDict.pkl"

            isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))

            df = pd.DataFrame()

            for i in range(len(isri_dataset['Jobdata'])):
                df_temp = pd.DataFrame.from_dict(isri_dataset['Jobdata'][i], orient='index')
                df = pd.concat([df, df_temp])

            df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10','time11','time12']] = pd.DataFrame(df.times.tolist(), index= df.index)
            df = df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10', 'time11', 'time12']]

            agglo_model = AgglomerativeClustering(n_clusters = n_cluster)

            y_means = agglo_model.fit_predict(df)

            #df.insert(0, "label", y_means, True)
            self.knn_model = KNeighborsRegressor(n_neighbors=3)
            self.knn_model.fit(df, y_means)
            with open(path+"knn_model_n"+str(n_cluster)+".pkl", "wb") as f:
                pickle.dump(self.knn_model, f)
            with open(path+"agglo_model_n"+str(n_cluster)+".pkl", "wb") as f:
                pickle.dump(self.knn_model, f)

    def label(self, data):
        data = np.array(data)
        data = data.reshape(1, -1)
        return self.knn_model.predict(data)