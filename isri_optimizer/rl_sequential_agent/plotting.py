import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator
from isri_optimizer.rl_sequential_agent.plotting_funktion import ergebnisse_plot, read_tensorflow_events, find_event_files_dict, ergebnisse_subplot

names = ['Kmeans', 'KNN', 'Ohne Cluster']
num_dict = {
    '_8': 'acht',
    '_12': 'zwölf',
    '_15': '15'
}
fenster = 31
linestyle = ['solid', 'solid', 'solid']
colors = ['tab:blue', 'tab:red', 'tab:green']
#base_directory = 'isri_optimizer/rl_sequential_agent/savefiles_Train1'
#event_file_paths = find_event_files_dict(base_directory) #erzeigt den dict Experimente

Experimente = {'sparse': {'_8': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_kmeans/3_sparse_8_kmeans_1/events.out.tfevents.1717196323.DESKTOP-6FHK9F7.8532.9', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_neighbour/3_sparse_8_neighbour_1/events.out.tfevents.1717202642.DESKTOP-6FHK9F7.8532.10', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_no_cluster/3_sparse_8_no_cluster_1/events.out.tfevents.1717205357.DESKTOP-6FHK9F7.8532.11'], 
            '_12': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_12_kmeans/3_sparse_12_kmeans_1/events.out.tfevents.1717207499.DESKTOP-6FHK9F7.8532.12', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_12_neighbour/3_sparse_12_neighbour_1/events.out.tfevents.1717213907.DESKTOP-6FHK9F7.8532.13', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_12_no_cluster/3_sparse_12_no_cluster_1/events.out.tfevents.1717216667.DESKTOP-6FHK9F7.8532.14'], 
            '_15': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_15_kmeans/3_sparse_15_kmeans_1/events.out.tfevents.1717218861.DESKTOP-6FHK9F7.8532.15', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_15_neighbour/3_sparse_15_neighbour_1/events.out.tfevents.1717225309.DESKTOP-6FHK9F7.8532.16', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_15_no_cluster/3_sparse_15_no_cluster_1/events.out.tfevents.1717228124.DESKTOP-6FHK9F7.8532.17']}, 
'dense': {'_8': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_8_kmeans/3_dense_8_kmeans_1/events.out.tfevents.1717276975.DESKTOP-6FHK9F7.2584.0', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_8_neighbour/3_dense_8_neighbour_1/events.out.tfevents.1717285774.DESKTOP-6FHK9F7.2584.2', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_8_no_cluster/3_dense_8_no_cluster_1/events.out.tfevents.1717283569.DESKTOP-6FHK9F7.2584.1'], 
          '_12': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_12_kmeans/3_dense_12_kmeans_1/events.out.tfevents.1717288546.DESKTOP-6FHK9F7.2584.3', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_12_neighbour/3_dense_12_neighbour_1/events.out.tfevents.1717297201.DESKTOP-6FHK9F7.2584.5', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_12_no_cluster/3_dense_12_no_cluster_1/events.out.tfevents.1717294960.DESKTOP-6FHK9F7.2584.4'], 
          '_15': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_15_kmeans/3_dense_15_kmeans_1/events.out.tfevents.1717300027.DESKTOP-6FHK9F7.2584.6', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_15_neighbour/3_dense_15_neighbour_1/events.out.tfevents.1717308779.DESKTOP-6FHK9F7.2584.8', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_dense_15_no_cluster/3_dense_15_no_cluster_1/events.out.tfevents.1717306509.DESKTOP-6FHK9F7.2584.7']}, 
'sparse_sum': {'_8': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_8_kmeans/3_sparse_sum_8_kmeans_1/events.out.tfevents.1717327636.DESKTOP-6FHK9F7.12692.0', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_8_neighbour/3_sparse_sum_8_neighbour_1/events.out.tfevents.1717336377.DESKTOP-6FHK9F7.12692.2', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_8_no_cluster/3_sparse_sum_8_no_cluster_1/events.out.tfevents.1717334152.DESKTOP-6FHK9F7.12692.1'], 
               '_12': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_12_kmeans/3_sparse_sum_12_kmeans_1/events.out.tfevents.1717339184.DESKTOP-6FHK9F7.12692.3', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_12_neighbour/3_sparse_sum_12_neighbour_1/events.out.tfevents.1717347924.DESKTOP-6FHK9F7.12692.5', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_12_no_cluster/3_sparse_sum_12_no_cluster_1/events.out.tfevents.1717345597.DESKTOP-6FHK9F7.12692.4'], 
               '_15': ['isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_15_kmeans/3_sparse_sum_15_kmeans_1/events.out.tfevents.1717350765.DESKTOP-6FHK9F7.12692.6', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_15_neighbour/3_sparse_sum_15_neighbour_1/events.out.tfevents.1717359505.DESKTOP-6FHK9F7.12692.8', 'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_sum_15_no_cluster/3_sparse_sum_15_no_cluster_1/events.out.tfevents.1717357214.DESKTOP-6FHK9F7.12692.7']}}

data_sparse_8 = read_tensorflow_events(Experimente, 'sparse', '_8')

normale_plots = False
subplots = True

for reward, values in Experimente.items():
    print(reward)
    for klassen, path in values.items():
        print(klassen)
        data = read_tensorflow_events(Experimente, reward, klassen)
        if normale_plots:
                ergebnisse_plot(data['x_rew'], data['y_rew'], labels=names, title=f'kummulierter Reward bei {num_dict[klassen]} Aktionen', file_name=f'Return{klassen}',
                        moving_average=True, ma_interval=fenster, line_styles=linestyle,
                        colors=colors, y_low=None, y_high=None,
                        x_label="Trainings-Iterationen", y_label="Return",
                        leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')

                ergebnisse_plot(data['x_diff'], data['y_diff'], labels=names, title=f'Vergleich der Auslastung bei {num_dict[klassen]} Aktionen', file_name=f'Diffsum{klassen}',
                        moving_average=True, ma_interval=fenster, line_styles=linestyle,
                        colors=colors, y_low=None, y_high=None,
                        x_label="Trainings-Iterationen", y_label="Relative Abweichung zur Baseline",
                        leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')

                ergebnisse_plot(data['x_tard'], data['y_tard'], labels=names, title=f'Vergleich der Termintreue bei {num_dict[klassen]} Aktionen', file_name=f'Tardiness{klassen}',
                        moving_average=True, ma_interval=fenster, line_styles=linestyle,
                        colors=colors, y_low=None, y_high=None,
                        x_label="Trainings-Iterationen", y_label="Relative Abweichung zur Baseline",
                        leg_pos='lower right', file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/')
        if subplots:
                titles = [
                    'kummulierter Reward',
                    'Vergleich der Auslastung',
                    'Vergleich der Termintreue',
                ]
                y_labels = ["Return", "Relative Abweichung zur Baseline", "Relative Abweichung zur Baseline"]
                ergebnisse_subplot([data['x_rew'], data['x_diff'], data['x_tard']], [data['y_rew'], data['y_diff'], data['y_tard']], 
                                    titles=titles, sup_title=f'Ergebnisse bei {num_dict[klassen]} Aktionen', x_label="Trainings-Iterationen", y_labels=y_labels, labels=names, moving_average=True, 
                                    ma_interval=fenster, line_styles=linestyle, colors=colors, y_low=None, y_high=None, leg_pos='lower right', 
                                    file_path=f'isri_optimizer/rl_sequential_agent/plots/{reward}/', file_name=f'subplots_{klassen}'
                                )



'''
ergebnisse_plot(x_rew, y_rew, labels=names, title='Return', file_name='Return',
                moving_average=True, ma_interval=fenster, leg_pos='lower right', line_styles=linestyle,
                colors=colors, y_low=None, y_high=None,
                x_label="Trainingsschritt", y_label="$\overline{R}$, gemittelter Return")

ergebnisse_plot(x_plan, y_plan, labels=names, title='Planerfüllung', file_name='Plan',
                moving_average=True, ma_interval=fenster, leg_pos='lower right',
                colors=colors, y_low=None, y_high=None, line_styles=linestyle,
                x_label="Trainingsschritt", y_label="$\omega$, Anteil richtig produzierter Produkte")

ergebnisse_plot(x1_0, y1_0, labels=names, title='Durchlaufzeiten', file_name='Durchlaufzeiten',
                moving_average=True, ma_interval=fenster, line_styles=linestyle,
                colors=colors, y_low=None, y_high=None,
                x_label="Trainingsschritt", y_label="Durchlaufzeit: Durchschnitt über alle Produkttypen [s]")

ergebnisse_plot(x_ausl, y_ausl, labels=names, title='Varianz der Auslastung', file_name='Auslastung',
                moving_average=True, ma_interval=fenster, y_scale='log',
                colors=colors, y_low=None, y_high=None, line_styles=linestyle,
                x_label="Trainingsschritt",
                y_label="$\sigma_{3}+\sigma_{4}$, Varianz der Auslastung in Abschnitt 3 und 4 ")
'''