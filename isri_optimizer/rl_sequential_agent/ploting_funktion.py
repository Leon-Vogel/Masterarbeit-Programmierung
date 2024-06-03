import matplotlib.pyplot as plt
import numpy as np
import os

def ergebnisse_plot(x_data_list, y_data_list, title="", x_label="", y_label="",
                    x_scale='linear', y_scale='linear',
                    x_ticks=None, y_ticks=None, font_size=7, file_path="isri_optimizer/rl_sequential_agent/plots/",
                    file_name="plot.png", dpi=500, figsize=(4, 3),
                    line_styles=None, colors=None, labels=None,
                    moving_average=False, ma_interval=1, leg_pos='upper right',
                    y_low=None, y_high=None):
    #plt.style.use('science')
    plt.rcParams.update({
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size
    })
    plt.rcParams['font.family'] = 'Times New Roman'
    '''plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times']})'''


    plt.figure(figsize=figsize)
    ax = plt.gca()

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

    plt.subplots_adjust(left=0.15, bottom=0.15)

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

    # Legende anzeigen
    if labels:
        plt.legend(loc=leg_pos, prop={'size': font_size})

    # Speichern des Plots
    plt.savefig(file_path + file_name + '.png', format='png', dpi=dpi)
    plt.savefig(file_path + file_name + '.svg', format='svg', dpi=dpi)

    # Schließe die Figur
    plt.close()


def find_event_files(base_dir):
    event_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                full_path = os.path.join(root, file)
                event_files.append(full_path)
    return event_files

# Beispielverwendung:
base_directory = 'isri_optimizer/rl_sequential_agent/savefiles_Train1'
event_file_paths = find_event_files(base_directory)

for path in event_file_paths:
    print(path)