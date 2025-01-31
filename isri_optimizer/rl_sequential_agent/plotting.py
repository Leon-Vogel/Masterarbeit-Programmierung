import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator
from plotting_funktion import ergebnisse_plot, read_tensorflow_events, find_event_files_dict, ergebnisse_subplot, ergebnisse_subplot_2x2

names = ['k-Means', 'Agglomerativ', 'Ohne Cluster']
num_dict = {
    '_8': 'acht',
    '_12': 'zwölf',
    '_15': '15'
}
fenster = 31
linestyle = ['solid', 'solid', 'solid']
colors = ['tab:blue', 'tab:red', 'tab:green']

Experimente = {'sparse': {'_8': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_kmeans/3_sparse_8_kmeans_1/events.out.tfevents.1717196323.DESKTOP-6FHK9F7.8532.9', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_neighbour/3_sparse_8_neighbour_1/events.out.tfevents.1717202642.DESKTOP-6FHK9F7.8532.10', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_no_cluster/3_sparse_8_no_cluster_1/events.out.tfevents.1717205357.DESKTOP-6FHK9F7.8532.11'], 
            '_12': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_12_kmeans/3_sparse_12_kmeans_1/events.out.tfevents.1717207499.DESKTOP-6FHK9F7.8532.12', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_12_neighbour/3_sparse_12_neighbour_1/events.out.tfevents.1717213907.DESKTOP-6FHK9F7.8532.13', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_12_no_cluster/3_sparse_12_no_cluster_1/events.out.tfevents.1717216667.DESKTOP-6FHK9F7.8532.14'], 
            '_15': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_15_kmeans/3_sparse_15_kmeans_1/events.out.tfevents.1717218861.DESKTOP-6FHK9F7.8532.15', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_15_neighbour/3_sparse_15_neighbour_1/events.out.tfevents.1717225309.DESKTOP-6FHK9F7.8532.16', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_15_no_cluster/3_sparse_15_no_cluster_1/events.out.tfevents.1717228124.DESKTOP-6FHK9F7.8532.17']}, 
'dense': {'_8': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_8_kmeans/3_dense_8_kmeans_1/events.out.tfevents.1717276975.DESKTOP-6FHK9F7.2584.0', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_8_neighbour/3_dense_8_neighbour_1/events.out.tfevents.1717285774.DESKTOP-6FHK9F7.2584.2', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_8_no_cluster/3_dense_8_no_cluster_1/events.out.tfevents.1717283569.DESKTOP-6FHK9F7.2584.1'], 
          '_12': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_12_kmeans/3_dense_12_kmeans_1/events.out.tfevents.1717288546.DESKTOP-6FHK9F7.2584.3', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_12_neighbour/3_dense_12_neighbour_1/events.out.tfevents.1717297201.DESKTOP-6FHK9F7.2584.5', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_12_no_cluster/3_dense_12_no_cluster_1/events.out.tfevents.1717294960.DESKTOP-6FHK9F7.2584.4'], 
          '_15': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_15_kmeans/3_dense_15_kmeans_1/events.out.tfevents.1717300027.DESKTOP-6FHK9F7.2584.6', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_15_neighbour/3_dense_15_neighbour_1/events.out.tfevents.1717308779.DESKTOP-6FHK9F7.2584.8', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_15_no_cluster/3_dense_15_no_cluster_1/events.out.tfevents.1717306509.DESKTOP-6FHK9F7.2584.7']}, 
'sparse_sum': {'_8': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_8_kmeans/3_sparse_sum_8_kmeans_1/events.out.tfevents.1717327636.DESKTOP-6FHK9F7.12692.0', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_8_neighbour/3_sparse_sum_8_neighbour_1/events.out.tfevents.1717336377.DESKTOP-6FHK9F7.12692.2', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_8_no_cluster/3_sparse_sum_8_no_cluster_1/events.out.tfevents.1717334152.DESKTOP-6FHK9F7.12692.1'], 
               '_12': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_12_kmeans/3_sparse_sum_12_kmeans_1/events.out.tfevents.1717339184.DESKTOP-6FHK9F7.12692.3', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_12_neighbour/3_sparse_sum_12_neighbour_1/events.out.tfevents.1717347924.DESKTOP-6FHK9F7.12692.5', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_12_no_cluster/3_sparse_sum_12_no_cluster_1/events.out.tfevents.1717345597.DESKTOP-6FHK9F7.12692.4'], 
               '_15': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_15_kmeans/3_sparse_sum_15_kmeans_1/events.out.tfevents.1717350765.DESKTOP-6FHK9F7.12692.6', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_15_neighbour/3_sparse_sum_15_neighbour_1/events.out.tfevents.1717359505.DESKTOP-6FHK9F7.12692.8', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_15_no_cluster/3_sparse_sum_15_no_cluster_1/events.out.tfevents.1717357214.DESKTOP-6FHK9F7.12692.7']}}

base_directory = 'isri_optimizer/rl_sequential_agent/savefiles_Train1'
Experimente = find_event_files_dict(base_directory) #erzeigt den dict Experimente
results_times = []
normale_plots = False
explained_variance = False
subplots = False
subplots_4 = True
for reward, values in Experimente.items():
    print(reward)
    for klassen, path in values.items():
        print(klassen)
        data = read_tensorflow_events(Experimente, reward, klassen)
        zeit_erfassen = False
        if zeit_erfassen:
            name = 0
            for i, total_time in enumerate(data['total_time']):
                if name > 2:
                     name = 0
                results_times.append({
                    'reward': reward,
                    'klassen': klassen,
                    'cluster_method': names[name],
                    'total_time': total_time
                })
                name += 1
        if normale_plots:
                ergebnisse_plot(data['x_rew'], data['y_rew'], labels=names, title=f'kummulierter Reward bei {num_dict[klassen]} Aktionen', 
                                file_name=f'Return{klassen}',moving_average=True, ma_interval=fenster, line_styles=linestyle,
                                colors=colors, y_low=None, y_high=None,x_label="Trainings-Iterationen", y_label="Return",
                                leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')

                ergebnisse_plot(data['x_diff'], data['y_diff'], labels=names, title=f'Vergleich der Auslastung bei {num_dict[klassen]} Aktionen', 
                                file_name=f'Diffsum{klassen}',moving_average=True, ma_interval=fenster, line_styles=linestyle,
                                colors=colors, y_low=None, y_high=None, x_label="Trainings-Iterationen", y_label="Relative Abweichung zur Baseline",
                                leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')

                ergebnisse_plot(data['x_tard'], data['y_tard'], labels=names, title=f'Vergleich der Termintreue bei {num_dict[klassen]} Aktionen', 
                                file_name=f'Tardiness{klassen}',moving_average=True, ma_interval=fenster, line_styles=linestyle,
                                colors=colors, y_low=None, y_high=None,x_label="Trainings-Iterationen", y_label="Relative Abweichung zur Baseline",
                                leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')

                ergebnisse_plot(data['x_diff_rew'], data['y_diff_rew'], labels=names, title=f'kummulierter Reward für Auslastung bei {num_dict[klassen]} Aktionen', 
                                file_name=f'Diffsum{klassen}_rew', moving_average=True, ma_interval=fenster, line_styles=linestyle,
                                colors=colors, y_low=None, y_high=None, x_label="Trainings-Iterationen", y_label="Reward für die Auslastung",
                                leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')

                ergebnisse_plot(data['x_tard_rew'], data['y_tard_rew'], labels=names, title=f'kummulierter Reward für Termintreue bei {num_dict[klassen]} Aktionen', 
                                file_name=f'Tardiness{klassen}_rew', moving_average=True, ma_interval=fenster, line_styles=linestyle,
                                colors=colors, y_low=None, y_high=None, x_label="Trainings-Iterationen", y_label="Reward für die Termintreue",
                                leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')
        if explained_variance:
            ergebnisse_plot(data['x_expl_var'], data['y_expl_var'], labels=names, title=f'Relativer Fehler der Value Funktion bei {num_dict[klassen]} Aktionen', 
                            file_name=f'Explained_Variance{klassen}',moving_average=True, ma_interval=fenster, line_styles=linestyle,
                                colors=colors, y_low=None, y_high=None,x_label="Trainings-Iterationen", y_label="Explained Variance",
                                leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/', y_top=1)

        if subplots:
                titles = [
                    'Kummulierter Reward',
                    'Vergleich der Auslastung',
                    'Vergleich der Termintreue',
                ]
                y_labels = ["Return", "Relative Abweichung zur Baseline", "Relative Abweichung zur Baseline"]
                ergebnisse_subplot([data['x_rew'], data['x_diff'], data['x_tard']], [data['y_rew'], data['y_diff'], data['y_tard']], 
                                    titles=titles, sup_title=f'Ergebnisse bei {num_dict[klassen]} Aktionen', x_label="Trainings-Iterationen", 
                                    y_labels=y_labels, labels=names, moving_average=True, 
                                    ma_interval=fenster, line_styles=linestyle, colors=colors, y_low=None, y_high=None, leg_pos='lower right', 
                                    file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/', file_name=f'subplots{klassen}'
                                )
        if subplots_4:
                titles = [
                    'Return',
                    'Vergleich der Belastung',
                    'Vergleich der Terminabweichung',
                    'Explained Variance'
                ]
                y_labels = ["Return", "Belastung relativ zum GA", "Verspätung relativ zum GA", "Explained Variance"]
                ergebnisse_subplot_2x2([data['x_rew'], data['x_diff'], data['x_tard'], data['x_expl_var']], [data['y_rew'], data['y_diff'], data['y_tard'], data['y_expl_var']], 
                                    titles=titles, show_titles=False, show_sup_title=False, sup_title=f'Ergebnisse bei {num_dict[klassen]} Aktionen', x_label="Trainingsiteration", 
                                    y_labels=y_labels, labels=names, moving_average=True, 
                                    ma_interval=fenster, line_styles=linestyle, colors=colors, y_low=None, y_high=None, leg_pos='lower right', 
                                    file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/{reward}', file_name=f'_subplots4_{klassen}'
                                )



if zeit_erfassen:
    Reward_Namen = ['Häufig', 'Selten, relativ zum GA', 'Selten']
    df = pd.DataFrame(results_times)
    mean_times = df.groupby(['reward', 'cluster_method'])['total_time'].mean().unstack()
    mean_times = mean_times / 60.0  
    mean_times = mean_times[['k-Means', 'Agglomerativ', 'Ohne Cluster']] 
    overall_mean_times = mean_times.mean()
    latex_table = (
        "\\begin{table}[ht]\n"
        "\\caption{Durchschnittliche Trainingsdauer (1,5 Millionen Schritte) für die Clustering und Reward Varianten}\n"
        "\\centering\n"
        "\\label{tab:zeiten_training}\n"
        "\\begin{tabular}{lccc}\n"
        "\\hline\n"
        " & \\textbf{Kmeans} & \\textbf{Agglomerativ} & \\textbf{Ohne Cluster} \\\\\n"
        "\\hline\n"
    )
    latex_table += (
        f"Trainingsdauer & "
        f"{overall_mean_times['Kmeans']:.1f} [min] & "
        f"{overall_mean_times['Agglomerativ']:.1f} [min] & "
        f"{overall_mean_times['Ohne Cluster']:.1f} [min] \\\\\n"
    )
    latex_table += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open('isri_optimizer/rl_sequential_agent/plots/average_training_times.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

