import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from data_preprocessing import IsriDataset


def load_isri_dataset_to_dataframe(path):
    isri_dataset = pickle.load(open(path, 'rb'))
    df = pd.concat([pd.DataFrame.from_dict(job, orient='index') for job in isri_dataset.data['Jobdata']], ignore_index=True)
    df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10', 'time11', 'time12']] = pd.DataFrame(df.times.tolist(), index=df.index)
    df_cluster = df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10', 'time11', 'time12']]
    return df, df_cluster


def calculate_inertia_and_distortion(df_cluster, k_range, cluster_algorithm):
    distortions = []
    inertias = []
    for k in k_range:
        model = cluster_algorithm(n_clusters=k).fit(df_cluster)
        distortions.append(sum(np.min(cdist(df_cluster, model.cluster_centers_, 'euclidean'), axis=1)) / df_cluster.shape[0])
        inertias.append(model.inertia_)
    return distortions, inertias


def plot_elbow_method(k_range, values, xlabel, ylabel, title, file_path, file_name, file_suffix, font_size=7, dpi=500, fig_size=None):
    if fig_size is None:
        fig_size = [4, 3]
    plt.rcParams.update({
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'font.family': 'Times New Roman',
        'figure.dpi': dpi,
        'figure.figsize': fig_size
    })
    plt.plot(k_range, values, 'bx-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, pad=20)
    plt.xticks(range(min(k_range), max(k_range)+1, 2))
    plt.grid(True)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.png", format='png', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.svg", format='svg', dpi=dpi)
    plt.close()


GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl"
df, df_cluster = load_isri_dataset_to_dataframe(GA_SOLUTIONS_PATH)
K = range(2, 21)

# Beispiel mit KMeans
distortions, inertias = calculate_inertia_and_distortion(df_cluster, K, KMeans)
plot_elbow_method(K, distortions, 'Anzahl der Klassen', 'Verzerrung', 'Die Ellenbogen-Methode mit Verzerrung für Kmeans-Clustering', 
                  'isri_optimizer/rl_sequential_agent/plots/data/', 'kmeans_elbow', 'distortion')
plot_elbow_method(K, inertias, 'Anzahl der Klassen', 'Inertia', 'Die Ellenbogen-Methode mit Inertia für Kmeans-Clustering', 
                  'isri_optimizer/rl_sequential_agent/plots/data/', 'kmeans_elbow', 'inertia')


