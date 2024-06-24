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
    if cluster == 'k-Means':
        cluster = 'Kmeans'
    distortions = []
    inertias = []
    silhouette = []
    davies_bouldin = []
    for k in k_range:
        print(k)
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


def plot_elbow_method(k_range, values, xlabel, ylabel, title, file_path, file_name, file_suffix, font_size=8, dpi=500, fig_size=None):
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
    plt.plot(k_range, values, 'b-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title, fontsize=font_size, pad=10)
    plt.xticks(range(min(k_range), max(k_range)+1, 2))
    plt.grid(True)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.png", format='png', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.svg", format='svg', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.pdf", format='pdf', dpi=dpi)
    plt.close()


def plot_elbow_method_multi(x_data_list, y_data_list, titles, sup_title, x_label, y_labels, 
                            file_path, file_name, file_suffix, 
                            show_sup_title=True, show_titles=True,
                            x_scale='linear', y_scale='linear', line_style=None,
                            font_size=8, dpi=500, fig_size=None):
    if fig_size is None:
        figsize=(4.5, 4)
    plt.rcParams.update({
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'font.family': 'Times New Roman'
    })
    
    subplot_labels = ['a)', 'b)', 'c)', 'd)']
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for plot_n, (ax, x_data, y_data, title, y_label, subplot_label) in enumerate(zip(axs.flatten(), x_data_list, y_data_list, titles, y_labels, subplot_labels), start=1):
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.grid(True)
        ax.set_xticks(range(min(x_data), max(x_data)+1))
        #ax.set_xticks(range(min(x_data), max(x_data)+1, 2))
        if show_titles:
            ax.set_title(title, fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        ax.text(-0.2, -0.1, subplot_label, transform=ax.transAxes, fontsize=font_size, verticalalignment='bottom', horizontalalignment='left')

        ax.plot(x_data, y_data, 'b-')
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % 2 != 0:
                label.set_visible(False)
    for ax in axs[1,:]:
        ax.set_xlabel(x_label, fontsize=font_size)
    if show_sup_title:
        fig.suptitle(sup_title, fontsize=font_size)
        plt.subplots_adjust(top=0.9)

    plt.tight_layout()

    plt.savefig(f"{file_path}{file_name}_{file_suffix}.png", format='png', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.svg", format='svg', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.pdf", format='pdf', dpi=dpi)
    plt.close()
        

def plot_hist(all_labels, bins, xlabel, ylabel, title, file_path, file_name, file_suffix, font_size=8, dpi=500, fig_size=None, grid=True):
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
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.pdf", format='pdf', dpi=dpi)
    plt.close()

def plot_relative_hist(all_labels, bins, xlabel, ylabel, title, file_path, file_name, file_suffix, font_size=8, dpi=500, fig_size=None, grid=True):
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
    
    # Berechne die relativen Häufigkeiten
    counts, bin_edges = np.histogram(all_labels, bins=np.arange(0.5, bins + 1.5, 1))
    relative_freqs = counts / len(all_labels)
    
    plt.hist(all_labels, bins=np.arange(0.5, bins + 1.5, 1), weights=np.ones_like(all_labels) / len(all_labels), edgecolor='black')
    
    plt.xticks(np.arange(1, bins+1, 1))  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=font_size)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    if grid:
        ax = plt.gca()
        ax.grid(True)
        ax.set_axisbelow(True)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.png", format='png', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.svg", format='svg', dpi=dpi)
    plt.savefig(f"{file_path}{file_name}_{file_suffix}.pdf", format='pdf', dpi=dpi)
    plt.close()

def plot_relative_hist_3x2(ax, all_labels, bins, xlabel, ylabel, title, font_size=8, grid=True, every_nth=1):
    all_labels = [x + 1 for x in all_labels]
    plt.rcParams.update({
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'font.family': 'Times New Roman'
    })
    # Berechne die relativen Häufigkeiten
    counts, bin_edges = np.histogram(all_labels, bins=np.arange(0.5, bins + 1.5, 1))
    relative_freqs = counts / len(all_labels)
    
    ax.hist(all_labels, bins=np.arange(0.5, bins + 1.5, 1), weights=np.ones_like(all_labels) / len(all_labels), edgecolor='black')
    ax.set_xticks(np.arange(1, bins+1, 1))
    if xlabel !='':
        ax.set_xlabel(xlabel)
    if col == 0:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=font_size)
    if grid:
        ax.grid(True)
        ax.set_axisbelow(True)
    if every_nth > 1:
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

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
    Achsen = ['Varianz innerhalb der Cluster', 'Inertia', 'Silhouettenkoeffizient', 'Davies-Bouldin Index']
    datei_endung = ['distortion', 'inertia', 'silhouette', 'davies_bouldin']
    Methoden = ['k-Means', 'Agglomeratives']
    for Methode in Methoden:
        distortions, inertias, silhouette, davies_bouldin = cluster_size_metrics(df_cluster, K, Methode)
        data = [distortions, inertias, silhouette, davies_bouldin]
        if Methode == 'k-Means':
            kmeans_data = data
        Titel = [f'Die Ellenbogen-Methode mit Distortion für {Methode} Clustering', f'Die Ellenbogen-Methode mit Inertia für {Methode} Clustering', 
            f'Der Silhouette Score für {Methode} Clustering', f'Der Davies Bouldin Score für {Methode} Clustering']
        for i in range(len(data)):
            plot_elbow_method(K, data[i], 'Anzahl der Klassen', Achsen[i], Titel[i], 
                        'isri_optimizer/rl_sequential_agent/plots/data/', Methode, datei_endung[i])
    titel_multi = ['k-Means Clustering', 'k-Means Clustering', 'Agglomeratives Clustering', 'Agglomeratives Clustering']
    plot_elbow_method_multi([K, K, K, K], [kmeans_data[0], kmeans_data[2], data[0], data[2]], 
                            titles=titel_multi, sup_title=None, 
                            x_label='Anzahl der Klassen', y_labels=['Varianz innerhalb der Cluster', 'Silhouettenkoeffizient', 'Varianz innerhalb der Cluster', 'Silhouettenkoeffizient'],
                            file_path='isri_optimizer/rl_sequential_agent/plots/data/', file_name='Cluster_Kennzahlen_Multiplot', 
                            file_suffix='2x2', show_sup_title=False, show_titles=True
    )

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
                #plot_hist(all_labels, i, 'Klassen', 'Anzahl Produkte einer Klasse', f'Häufigkeit der Klassen für {Namen[counter]} Clustering', 
                #          'isri_optimizer/rl_sequential_agent/plots/data/count/', f'{Methode}_Count', f'n{i}')
                plot_relative_hist(all_labels, i, 'Klassen', 'relative Häufigkeit', title=f'{Namen[counter]} Clustering', 
                          file_path='isri_optimizer/rl_sequential_agent/plots/data/count/', file_name=f'{Namen[counter]}_Count', file_suffix=f'n{i}')

plot_cluster_count_3x2 = False
if plot_cluster_count_3x2:
    GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl"
    path="./isri_optimizer/rl_sequential_agent/cluster_models/"
    n_cluster = [8, 12, 15]
    isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
    Methoden = ['kmeans', 'knn']
    Namen = ['k-Means', 'Agglomeratives']
    plt.rcParams.update({
        'font.size': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'font.family': 'Times New Roman'
    })
    fig, axs = plt.subplots(len(n_cluster), len(Methoden), figsize=(4.5, 5.5))
    #fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for row, i in enumerate(n_cluster):
        for col, Methode in enumerate(Methoden):
            if os.path.exists(f"{path}{Methode}_model_n{i}.pkl"):
                with open(f"{path}{Methode}_model_n{i}.pkl", "rb") as f:
                    model = pickle.load(f)
                all_labels = []
                for test_id in range(606):
                    times_array = [j['times'] for j in isri_dataset.data['Jobdata'][test_id].values()]
                    labels = model.predict(times_array)
                    all_labels.append(labels)
                all_labels = np.array(all_labels).flatten()
                
                ax = axs[row, col]
                if row == 0:
                    plot_relative_hist_3x2(ax, all_labels, i, '', 'relative Häufigkeit', title=f'{Namen[col]} Clustering')
                elif row == 2:
                    plot_relative_hist_3x2(ax, all_labels, i, 'Klasse', 'relative Häufigkeit', title='', every_nth=2)
                else:
                    plot_relative_hist_3x2(ax, all_labels, i, '', 'relative Häufigkeit', title='')
    plt.tight_layout()
    plt.savefig('isri_optimizer/rl_sequential_agent/plots/data/count/cluster_histograms.png', format='png', dpi=500)
    plt.savefig('isri_optimizer/rl_sequential_agent/plots/data/count/cluster_histograms.svg', format='svg', dpi=500)
    plt.savefig('isri_optimizer/rl_sequential_agent/plots/data/count/cluster_histograms.pdf', format='pdf', dpi=500)
    plt.close()


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
    #create_histogram_data_table(data, 'den genetischen Algorithmus', 'isri_optimizer/rl_sequential_agent/plots/data/histogram_data.tex')
    #plot_hist(data, np.amax(data)+1, 'n dringendster Auftrag', 'Häufigkeit', 'Auswahl der n dringendsten Aufträge bei dem genetischen Algorithmus', 
    #                      'isri_optimizer/rl_sequential_agent/plots/data/', 'GA', 'Dringlichkeit')
    plot_relative_hist(data, np.amax(data)+1, 'Priorität nach Deadline', 'relative Häufigkeit', title=None, #'Auswahl der n dringendsten Aufträge bei dem genetischen Algorithmus'
                          file_path='isri_optimizer/rl_sequential_agent/plots/data/', file_name='GA', file_suffix='Dringlichkeit_relativ')
    






























