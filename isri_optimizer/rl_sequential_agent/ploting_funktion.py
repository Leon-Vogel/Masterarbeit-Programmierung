import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd

def ergebnisse_plot(x_data_list, y_data_list, title="", x_label="", y_label="",
                    x_scale='linear', y_scale='linear',
                    x_ticks=None, y_ticks=None, font_size=7, file_path="isri_optimizer/rl_sequential_agent/plots/",
                    file_name="plot.png", dpi=500, figsize=(4, 3),
                    line_styles=None, colors=None, labels=None,
                    moving_average=False, ma_interval=1, leg_pos='upper right',
                    y_low=None, y_high=None):

    plt.rcParams.update({
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'font.family': 'Times New Roman'
    })
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    

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

    plt.subplots_adjust(left=0.15, bottom=0.15)
    ax.grid(True)

    # Plotten der Datenreihen
    if line_styles is None:
        line_styles = ['-'] * len(x_data_list)

    if colors is None:
        colors = [None] * len(x_data_list)

    log = []
    for i, (x_data, y_data) in enumerate(zip(x_data_list, y_data_list)):
        label = labels[i] if labels else f"Datenreihe {i + 1}"
        #plt.plot(x_data, y_data, label=label, linestyle=line_styles[i], color=colors[i])

        # Plot des gleitenden Durchschnitts (falls aktiviert)
        if moving_average and ma_interval > 1 and len(y_data) >= ma_interval:
            moving_avg = np.convolve(y_data, np.ones(ma_interval) / ma_interval, mode='valid')
            ma_x_data = x_data[ma_interval // 2:-(ma_interval // 2)]
            plt.plot(ma_x_data, moving_avg, label=label, linestyle=line_styles[i],
                     color=colors[i], alpha=0.7)
            log.append(moving_avg)
            
    if y_low is not None or y_high is not None:
        ax.set_ylim(y_low, y_high)
    else:
        current_y_limits = ax.get_ylim()
        if current_y_limits[1] < 0:
            ax.set_ylim(bottom=current_y_limits[0], top=0)
    # Legende anzeigen
    if labels:
        plt.legend(loc=leg_pos, prop={'size': font_size})

    # Speichern des Plots
    plt.savefig(file_path + file_name + '.png', format='png', dpi=dpi)
    plt.savefig(file_path + file_name + '.svg', format='svg', dpi=dpi)
    plt.close()



def find_event_files(base_dir):
    event_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                full_path = os.path.join(root, file)
                normalized_path = os.path.normpath(full_path).replace(os.sep, '/')
                event_files.append(normalized_path)
    return event_files


def find_event_files_dict(base_dir):
    event_files = {
        'sparse': {'_8': [], '_12': [], '_15': []},
        'dense': {'_8': [], '_12': [], '_15': []},
        'sparse_sum': {'_8': [], '_12': [], '_15': []}
    }

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                full_path = os.path.join(root, file)
                # Normalisiere den Pfad und ersetze \ durch /
                normalized_path = os.path.normpath(full_path).replace(os.sep, '/')

                # Finde das Stichwort und die Zahl im Pfad in einer bestimmten Reihenfolge
                if 'sparse_sum' in normalized_path:
                    for suffix in event_files['sparse_sum']:
                        if suffix in normalized_path:
                            event_files['sparse_sum'][suffix].append(normalized_path)
                            break
                elif 'sparse' in normalized_path:
                    for suffix in event_files['sparse']:
                        if suffix in normalized_path:
                            event_files['sparse'][suffix].append(normalized_path)
                            break
                elif 'dense' in normalized_path:
                    for suffix in event_files['dense']:
                        if suffix in normalized_path:
                            event_files['dense'][suffix].append(normalized_path)
                            break

    return event_files

def read_tensorflow_events(event_files, keyword, suffix):
    data = {
        'x_rew': [],
        'y_rew': [],
        'x_diff': [],
        'y_diff': [],
        'x_tard': [],
        'y_tard': []
    }

    files = event_files.get(keyword, {}).get(suffix, [])
    for file in files:
        d = {}
        for event in summary_iterator(file):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    if value.tag in d.keys():
                        d[value.tag].append(value.simple_value)
                    else:
                        d.update({str(value.tag): [value.simple_value]})
        df = pd.DataFrame.from_dict(d, orient='index').transpose()
        
        if 'rollout/ep_rew_mean' in df.columns:
            data['x_rew'].append(list(df['rollout/ep_rew_mean'].index.values))
            data['y_rew'].append(df['rollout/ep_rew_mean'].to_list())
        
        if 'workload_gap' in df.columns:
            data['x_diff'].append(list(df['workload_gap'].index.values))
            data['y_diff'].append(df['workload_gap'].to_list())
        
        if 'deadline_gap' in df.columns:
            data['x_tard'].append(list(df['deadline_gap'].index.values))
            data['y_tard'].append(df['deadline_gap'].to_list())
    
    return data