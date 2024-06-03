import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator
from ploting_funktion import ergebnisse_plot

fsize = 7
tsize = 7
tdir = 'in'
major = 5
minor = 3
style = 'default'  # 'default' helvetica


def ergebnisse_plot_alt(x_data_list, y_data_list, title="", x_label="", y_label="",
                    x_scale='linear', y_scale='linear',
                    x_ticks=None, y_ticks=None, font_size=7, file_path="isri_optimizer/rl_sequential_agent/plots/",
                    file_name="plot.png", dpi=500, figsize=(4, 3),
                    line_styles=None, colors=None, labels=None,
                    moving_average=False, ma_interval=1, leg_pos='upper right',
                    y_low=None, y_high=None):
    """
    Erstellt einen Plot und speichert ihn in einem Dateiformat.

    Parameter:
        x_data_list (list of lists): Die Liste von x-Werten für jede Datenreihe.
        y_data_list (list of lists): Die Liste von y-Werten für jede Datenreihe.
        title (str): Der Titel des Plots (Standard: leer).
        x_label (str): Die Beschriftung der x-Achse (Standard: leer).
        y_label (str): Die Beschriftung der y-Achse (Standard: leer).
        x_scale (str): Die Skala der x-Achse (Standard: 'linear').
        y_scale (str): Die Skala der y-Achse (Standard: 'linear').
        x_ticks (list): Benutzerdefinierte x-Ticks (Standard: None).
        y_ticks (list): Benutzerdefinierte y-Ticks (Standard: None).
        font_size (int): Die Schriftgröße für Titel und Achsenbeschriftungen (Standard: 12).
        file_path (str): Der Ordner, in dem der Plot gespeichert werden soll (Standard: '').
        file_name (str): Der Dateiname, unter dem der Plot gespeichert werden soll (Standard: 'plot.png').
        dpi (int): Die Auflösung des gespeicherten Plots in DPI (Standard: 300).
        figsize (tuple): Die Größe des Plots in Zoll (Standard: (8, 6)).
        line_styles (list): Die Linienstile für jede Datenreihe (Standard: None).
        colors (list): Die Farben für jede Datenreihe (Standard: None).
        labels (list): Die Legendenbeschriftungen für jede Datenreihe (Standard: None).
        moving_average (bool): Gibt an, ob der gleitende Durchschnitt dargestellt werden soll (Standard: False).
        ma_interval (int): Das Intervall für den gleitenden Durchschnitt (Standard: 1, d.h. kein Durchschnitt).

    Rückgabe:
        None
    """
    # Erstelle eine neue Figur und Achsen
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times']})
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.labelsize'] = tsize
    plt.rcParams['ytick.labelsize'] = tsize
    # plt.rcParams['xtick.major.size'] = tsize
    # plt.rcParams['xtick.minor.size'] = tsize
    # plt.rcParams['ytick.major.size'] = major
    # plt.rcParams['ytick.minor.size'] = minor

    # Setze die Achsenskalen
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if y_low is not None:
        ax.set_ylim(y_low, y_high)

    # Benutzerdefinierte x-Ticks
    if x_ticks:
        ax.set_xticks(x_ticks)

    # Benutzerdefinierte y-Ticks
    if y_ticks:
        ax.set_yticks(y_ticks)

    # Setze die Schriftgröße für Titel und Achsenbeschriftungen
    plt.title(title, fontsize=font_size)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    # Plotten der Datenreihen
    if line_styles is None:
        line_styles = ['-'] * len(x_data_list)

    if colors is None:
        colors = [None] * len(x_data_list)

    log = []
    for i, (x_data, y_data) in enumerate(zip(x_data_list, y_data_list)):
        label = labels[i] if labels else f"Datenreihe {i + 1}"
        # plt.plot(x_data, y_data, label=label, linestyle=line_styles[i], color=colors[i])

        # Plot des gleitenden Durchschnitts (falls aktiviert)
        if moving_average and ma_interval > 1 and len(y_data) >= ma_interval:
            moving_avg = np.convolve(y_data, np.ones(ma_interval) / ma_interval, mode='valid')
            ma_x_data = x_data[ma_interval // 2:-(ma_interval // 2)]
            plt.plot(ma_x_data, moving_avg, label=label, linestyle=line_styles[i],
                     color=colors[i], alpha=0.7)
            log.append(moving_avg)

    #with open(file_path+file_name+'moving_avg.txt', 'w') as file:
    #    for data in log:
    #        file.write(str(data[-30:]) + '\n')

    # Legende anzeigen
    if labels:
        plt.legend(loc=leg_pos, prop={
            'family': 'Helvetica'})

    # plt.ylim([y_low, y_high])

    # Speichern des Plots
    plt.savefig(file_path + file_name + '.png', format='png', dpi=dpi)
    plt.savefig(file_path + file_name + '.svg', format='svg', dpi=dpi)
    # plt.savefig(file_path + file_name, format='png', dpi=dpi)

    # Zeige den Plot an (optional)
    # plt.show()

    # Schließe die Figur
    plt.close()


# Beispiel:


Experimente = [
    'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_kmeans/3_sparse_8_kmeans_1/events.out.tfevents.1717196323.DESKTOP-6FHK9F7.8532.9',
    'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_neighbour/3_sparse_8_neighbour_1/events.out.tfevents.1717202642.DESKTOP-6FHK9F7.8532.10',
    'isri_optimizer/rl_sequential_agent/savefiles_Train1/_3_sparse_8_no_cluster/3_sparse_8_no_cluster_1/events.out.tfevents.1717205357.DESKTOP-6FHK9F7.8532.11']

#names = [r'$Kmeans$', r'$KNN$', r'$No Cluster$']
names = [r'Kmeans', r'KNN', r'No Cluster']
x_rew = []
y_rew = []
x_diff = []
y_diff = []
x_tard = []
y_tard = []


for i in range(len(names)):
    d = {}
    for event in summary_iterator(Experimente[i]):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag in d.keys():
                    d[value.tag].append(value.simple_value)
                else:
                    d.update({str(value.tag): [value.simple_value]})
                    # print(value.simple_value)
    # print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.transpose()

    x_rew.append(list(df['rollout/ep_rew_mean'].index.values))
    y_rew.append(df['rollout/ep_rew_mean'].to_list())
    x_diff.append(list(df['workload_gap'].index.values))
    y_diff.append(df['workload_gap'].to_list())
    x_tard.append(list(df['deadline_gap'].index.values))
    y_tard.append(df['deadline_gap'].to_list())
    '''

    x1_0.append(df['Dlz/Typ1'].index.values)
    y1_1 = (df['Dlz/Typ1'].to_list())
    y1_2 = (df['Dlz/Typ2'].to_list())
    y1_3 = (df['Dlz/Typ3'].to_list())
    y1_4 = (df['Dlz/Typ4'].to_list())
    y1_5 = (df['Dlz/Typ5'].to_list())
    tmp = np.array([y1_1, y1_2, y1_3, y1_4, y1_5])
    y1_0.append(np.average(tmp, axis=0))'''

# print(x_rew)
# print(y_rew)


# names = ['$RAM$', '$RWM$', '$RAP$', '$RWP$']

fenster = 31
linestyle = ['solid', 'solid', 'solid']
colors = ['tab:blue', 'tab:red', 'tab:green']


ergebnisse_plot(x_rew, y_rew, labels=names, title='Return bei acht Aktionen', file_name='Return8',
                moving_average=True, ma_interval=fenster, line_styles=linestyle,
                colors=colors, y_low=None, y_high=None,
                x_label="Rollout", y_label="gemittelter Return",
                leg_pos='lower right', file_path="isri_optimizer/rl_sequential_agent/plots/sparse/")


ergebnisse_plot(x_diff, y_diff, labels=names, title='Vergleich der Auslastung bei acht Aktionen', file_name='Diffsum8',
                moving_average=True, ma_interval=fenster, line_styles=linestyle,
                colors=colors, y_low=None, y_high=None,
                x_label="Rollout",
                y_label="Diffsum Unterschied zum GA",
                leg_pos='lower right', file_path="isri_optimizer/rl_sequential_agent/plots/sparse/")

ergebnisse_plot(x_tard, y_tard, labels=names, title='Vergleich der Termintreue bei acht Aktionen', file_name='Tardiness8',
                moving_average=True, ma_interval=fenster, line_styles=linestyle,
                colors=colors, y_low=None, y_high=None,
                x_label="Rollout",
                y_label="Tardiness Unterschied zum GA",
                leg_pos='lower right', file_path="isri_optimizer/rl_sequential_agent/plots/sparse/")


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