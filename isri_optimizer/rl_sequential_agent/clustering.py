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
import pickle
import os

class cluster_kmeans():
    def __init__(self, n_cluster, path="isri_optimizer/rl_sequential_agent/"):
        if os.path.exists(path+"kmeans_model.pkl"):
            with open(path+"kmeans_model.pkl", "rb") as f:
                self.kmeans = pickle.load(f)
        else:
            GA_SOLUTIONS_PATH = "isri_optimizer/rl_sequential_agent/IsriDataDict.pkl"

            isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))

            df = pd.DataFrame()

            for i in range(len(isri_dataset['Jobdata'])):
                df_temp = pd.DataFrame.from_dict(isri_dataset['Jobdata'][i], orient='index')
                df = pd.concat([df, df_temp])

            df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10','time11','time12']] = pd.DataFrame(df.times.tolist(), index= df.index)
            df = df[['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10', 'time11', 'time12']]

            self.kmeans = KMeans(n_clusters=n_cluster).fit(df)
            with open(path+"kmeans_model.pkl", "wb") as f:
                pickle.dump(self.kmeans, f)
        
    def label(self, data):
        labels = self.kmeans.predict(data)
        return labels



#Test:
Cluster = cluster_kmeans(8)

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
print(y)