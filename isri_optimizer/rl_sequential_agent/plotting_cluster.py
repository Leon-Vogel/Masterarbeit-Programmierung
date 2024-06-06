import copy
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from data_preprocessing import IsriDataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestCentroid
import os


def load_isri_dataset_to_dataframe(path):
    isri_dataset = pickle.load(open(path, 'rb'))
    df = pd.concat([pd.DataFrame.from_dict(job, orient='index') for job in isri_dataset.data['Jobdata']], ignore_index=True)
    df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10', 'time11', 'time12']] = pd.DataFrame(df.times.tolist(), index=df.index)
    df_cluster = df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10', 'time11', 'time12']]
    return df, df_cluster

def cluster_size_metrics(df_cluster, k_range, cluster):
    distortions = []
    inertias = []
    silhouette = []
    davies_bouldin = []
    for k in k_range:
        if cluster == 'Kmeans':
            model = KMeans(n_clusters=k).fit(df_cluster)
            y_means = model.predict(df_cluster)
            silhouette.append(silhouette_score(df_cluster, y_means))
            davies_bouldin.append(davies_bouldin_score(df_cluster, y_means))
            distortions.append(sum(np.min(cdist(df_cluster, model.cluster_centers_, 'euclidean'), axis=1)) / df_cluster.shape[0])
            inertias.append(model.inertia_)
        elif cluster == 'Agglomeratives':
            model = AgglomerativeClustering(n_clusters = k)
            y_means = model.fit_predict(df_cluster)
            silhouette.append(silhouette_score(df_cluster, y_means))
            davies_bouldin.append(davies_bouldin_score(df_cluster, y_means))
            clf = NearestCentroid()
            clf.fit(df_cluster, y_means)
            distortions.append(sum(np.min(cdist(df_cluster, clf.centroids_, 'euclidean'), axis=1)) / df_cluster.shape[0])
            inertia = sum(np.linalg.norm(df_cluster[model.labels_ == i] - clf.centroids_[i]) ** 2 for i in range(k))
            inertias.append(inertia)
    return distortions, inertias, silhouette, davies_bouldin


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
    plt.title(title, fontsize=font_size, pad=10)
    plt.xticks(range(min(k_range), max(k_range)+1, 2))
    plt.grid(True)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.png", format='png', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.svg", format='svg', dpi=dpi)
    plt.close()

def plot_hist(all_labels, bins, xlabel, ylabel, title, file_path, file_name, file_suffix, font_size=7, dpi=500, fig_size=None, grid=True):
    all_labels = [x + 1 for x in all_labels]
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
    plt.figure(figsize=fig_size)
    plt.hist(all_labels, bins=np.arange(0.5, bins + 1.5, 1), edgecolor='black')
    
    plt.xticks(np.arange(1, bins+1, 1))  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=font_size) #, pad=10
    plt.subplots_adjust(left=0.15, bottom=0.15)
    if grid:
        ax = plt.gca()
        ax.grid(True)
        ax.set_axisbelow(True) 
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.png", format='png', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.svg", format='svg', dpi=dpi)
    plt.close()

def create_histogram_data_table(data, experiment_type, output_path):
    df = pd.DataFrame(data, columns=['Shift'])
    df['Shift'] = df['Shift'] + 1
    freq_table = df['Shift'].value_counts().reset_index()
    freq_table.columns = ['Shift', 'Frequency']
    freq_table = freq_table.sort_values(by='Shift').reset_index(drop=True)
    if len(freq_table) % 2 != 0:
        freq_table = freq_table._append({'Shift': '', 'Frequency': 0}, ignore_index=True)
    mid_index = len(freq_table) // 2
    first_half = freq_table.iloc[:mid_index]
    second_half = freq_table.iloc[mid_index:]
    transposed_first_half = first_half.set_index('Shift').T
    transposed_second_half = second_half.set_index('Shift').T
    transposed_second_half.columns = transposed_second_half.columns[:-1].tolist() + [18]
    table = (
        f"\\begin{{table}}[ht]\n\\centering\n"
        f"\\caption{{Verteilung der Dringlichkeit der ausgewählten Aufträge nach Liefertermin für {experiment_type}}}\n"
        "\\begin{tabular}{l"
    )
    table += "c" * len(transposed_first_half.columns) + "}\n"
    table += "\\hline\n"
    table += "\\textbf{Position der Dringlichkeit} & " + " & ".join(map(str, transposed_first_half.columns)) + " \\\\\n"
    table += "\\hline\n"
    for index, row in transposed_first_half.iterrows():
        table += "\\textbf{Häufigkeit der Auswahl} & " + " & ".join(map(str, row.values)) + " \\\\\n"
    table += "\\hline\n"
    table += "\\hline\n"
    table += "\\textbf{Position der Dringlichkeit} & " + " & ".join(map(str, transposed_second_half.columns)) + " \\\\\n"
    table += "\\hline\n"
    for index, row in transposed_second_half.iterrows():
        table += "\\textbf{Häufigkeit der Auswahl} & " + " & ".join(map(str, row.values)) + " \\\\\n"
    table += "\\hline\n"
    table += "\\end{tabular}\n\\end{table}\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(table)

plot_cluster_metrics = False
if plot_cluster_metrics:
    GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl"
    df, df_cluster = load_isri_dataset_to_dataframe(GA_SOLUTIONS_PATH)
    K = range(2, 21)
    Achsen = ['Distortion', 'Inertia', 'Silhouette Score', 'Davies Bouldin Score']
    datei_endung = ['distortion', 'inertia', 'silhouette', 'davies_bouldin']
    Methoden = ['Kmeans', 'Agglomeratives']
    for Methode in Methoden:
        distortions, inertias, silhouette, davies_bouldin = cluster_size_metrics(df_cluster, K, Methode)
        data = [distortions, inertias, silhouette, davies_bouldin]
        Titel = [f'Die Ellenbogen-Methode mit Distortion für {Methode} Clustering', f'Die Ellenbogen-Methode mit Inertia für {Methode} Clustering', 
            f'Der Silhouette Score für {Methode} Clustering', f'Der Davies Bouldin Score für {Methode} Clustering']
        for i in range(len(data)):
            plot_elbow_method(K, data[i], 'Anzahl der Klassen', Achsen[i], Titel[i], 
                        'isri_optimizer/rl_sequential_agent/plots/data/', Methode, datei_endung[i])

plot_cluster_count = False
if plot_cluster_count:
    GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl"
    path="./isri_optimizer/rl_sequential_agent/cluster_models/"
    n_cluster = [8, 12, 15]
    isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
    Methoden = ['kmeans', 'knn']
    Namen = ['Kmeans', 'Agglomeratives']
    for counter, Methode in enumerate(Methoden):
        for i in n_cluster:
            if os.path.exists(f"{path}{Methode}_model_n{i}.pkl"):
                with open(f"{path}{Methode}_model_n{i}.pkl", "rb") as f:
                    model = pickle.load(f)
                all_labels = []
                for test_id in range(606):
                    times_array = [j['times'] for j in isri_dataset.data['Jobdata'][test_id].values()]

                    labels = model.predict(times_array)
                    all_labels.append(labels)
                all_labels = np.array(all_labels).flatten()
                plot_hist(all_labels, i, 'Klassen', 'Anzahl Produkte einer Klasse', f'Häufigkeit der Klassen für {Namen[counter]} Clustering', 
                          'isri_optimizer/rl_sequential_agent/plots/data/count/', f'{Methode}_Count', f'n{i}')

plot_GA_solution = True
if plot_GA_solution:
    GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl"
    isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
    shift_record = pd.DataFrame()
    for i in range(len(isri_dataset.data['GAChromosome'])):
        keys_list = isri_dataset.data['GAChromosome'][i]
        keys_list = keys_list.tolist()
        data = copy.deepcopy(isri_dataset.data['Jobdata'][i])
        shifts = {}
        for key in keys_list:
            if key in data:
                original_index = list(data.keys()).index(key)
                shift = original_index
                shifts[key] = shift
                del data[key] 
        df = pd.DataFrame.from_dict(shifts, orient='index')
        shift_record = pd.concat([shift_record, df], ignore_index=True)
    data = shift_record.values.tolist()
    data = np.array(data).flatten()
    create_histogram_data_table(data, 'den genetischen Algorithmus', 'isri_optimizer/rl_sequential_agent/plots/data/histogram_data.tex')
    plot_hist(data, np.amax(data)+1, 'n dringendster Auftrag', 'Häufigkeit', 'Auswahl der n dringendsten Aufträge bei dem Genetischen Algorithmus', 
                          'isri_optimizer/rl_sequential_agent/plots/data/', 'GA', 'dringlichkeit')






























