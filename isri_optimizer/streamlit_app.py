from _dir_init import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
sys.path.append(os.path.join(os.path.abspath(''), "..")) # TODO: ist das austauschbar mit _dir_init?
from isri_on_premise import isri_optimize
from misc_utils.color_palette import colors
from isri_simulation.isri_simulation import Simulation, Siminfo
import copy
import streamlit as st
import pickle
st.set_page_config(layout="wide")
import configparser
import pyodbc
import json
from datetime import datetime


config = configparser.ConfigParser()
config.read('./isri_optimizer/config.ini')
#config.read('config.ini')
# DB hat jetzt XML- schwerer einzulesen deshalb wieder json

data_A = pd.read_json(config['GA']['savepath']+'_A')
data_B = pd.read_json(config['GA']['savepath']+'_B')

data = data_A
# Plot Funktionen

# @st.cache_resource
def plot_werkerlast_bar(data: np.ndarray, deadlines: list, title: str, _streamlit_element):
    fig = plt.figure(num=title, clear=True)
    ax = plt.subplot(111)
    colors = map_to_rgb(deadlines)
    ax.bar(np.arange(0, data.shape[0]), height=data, width=0.5, color=colors, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Produktreihenfolge')
    ax.set_ylabel('MTM Zeit pro Produkt')
    with _streamlit_element:
        st.pyplot(fig, use_container_width=True, clear_figure=True)

def map_to_rgb(array: np.ndarray):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    cmap = cm.get_cmap('RdYlGn')
    colors = cmap(normalized_array)
    return colors

# @st.cache_resource
def plot_opti_data(data, _streamlit_element):
    fig, axs = plt.subplots(2, 1, figsize=[5, 10])
    axs[0].set_title("Maximierung der Belastungs-Abwechslung")
    axs[0].plot(data['mean_workload_diff'], ls='-', label='Mean in Population')
    axs[0].plot(data['min_workload_diff'], ls='-', label='Minimum')
    axs[0].plot(data['std_workload_diff'], ls='-', label='Standard Deviation')
    axs[0].set_xlabel("Generation im genetischen Algorithmus")
    axs[0].set_ylabel("Summe der absoluten Differenzen")
    axs[0].legend()

    axs[1].set_title("Verspätung")
    axs[1].plot(data['mean_tardiness'], ls='-', label='Mean in Population')
    axs[1].plot(data['min_tardiness'], ls='-', label='Minimum')
    axs[1].plot(data['std_tardiness'], ls='-', label='Standard Deviation')
    axs[1].set_xlabel("Generation im genetischen Algorithmus")
    axs[1].set_ylabel("Exponentiell gewichtete Verspätung")
    axs[1].legend()
    
    with _streamlit_element:
        st.pyplot(fig, use_container_width=False)

def update_data(data, selected_number, target_index):
    row_to_move = data[data['auidnr'] == selected_number]
    # Dropping the row from its original position
    data = data.drop(row_to_move.index)
    # Re-inserting the row at the target index
    data = pd.concat([data.iloc[:target_index], row_to_move, data.iloc[target_index:]]).reset_index(drop=True)
    return data

def plot_dataframe(data):
    plot_frame = copy.deepcopy(data[['produktionsnummer', 'Deadline', 'auidnr']])
    op_cols = ['OP11', 'OP12', 'OP13', 'OP14', 'OP15', 'OP31', 'OP32', 'OP33', 'OP34', 'OP35', 'EOLMech', 'EOL_elektr.']
    plot_frame['MTM-Zeit'] = [[row[op_col] for op_col in op_cols] for idx, row in data.iterrows()]

    print("Updating Data Editor")
    tab_data.data_editor(
        plot_frame,
        column_config={
            "Deadline": st.column_config.TextColumn('Deadline', disabled=True),
            "MTM-Zeit": st.column_config.BarChartColumn(
                "Aufwand Pro Arbeitsplatz", y_min=0, y_max=250, width=400
            ),
        },
        hide_index=False,
        num_rows='fixed',
        height=2000
    )

# Tabs
tab_chart, tab_data, tab_optimization = st.tabs(["Plot", 'Daten', "Optimierung"])

# Control Panel
with st.sidebar:
    linie = st.selectbox('Linie: ', ['A', 'B'])
    goal = st.selectbox('Grafik: ', ['Werkerlast'])
    n_lines = st.number_input('Anzahl Linien', 1, 4, 2, 1)
    jpl = st.number_input("Produkte pro Linie", 1, 96, 36, 1)
    if st.button('Start Optimierung'):
        # Anpassen der Konfiguration
        config_path  = './isri_optimizer/config.ini'
        config = configparser.ConfigParser()
        config.read(config_path)
        config.set("GA", "n_lines", str(n_lines))
        config.set("GA", "jobs_per_line", str(jpl))
        fitness = isri_optimize(config)
        # Updata Data
        st.write(f"Verspätung: {fitness[1]}")
        st.write(f"Workload Differenzsumme: {fitness[0]}")
        data_A = pd.read_json(config['GA']['savepath']+'_A')
        data_B = pd.read_json(config['GA']['savepath']+'_B')

    auftrag = st.selectbox('Auftrag', data.auidnr)
    position = st.selectbox('Position', [str(idx) for idx in range(len(data_A) + 1)])
    if st.button("Position Ändern"):
        # plt.close('all')
        data = update_data(data, auftrag, int(position))
        plot_dataframe(data)
        # Werkerlast Plot
        lastplot_col1, lastplot_col2 = tab_chart.columns([0.5, 0.5])
        op_cols = ['OP11', 'OP12', 'OP13', 'OP14', 'OP15', 'OP31', 'OP32', 'OP33', 'OP34', 'OP35', 'EOLMech', 'EOL_elektr.']
        lastplot_containers = []
        for idx, op in enumerate(op_cols):
            container = lastplot_containers[idx]
            data_op = data[op].values
            deadlines = data['Deadline']
            plot_werkerlast_bar(data_op, deadlines, op, container)
    
    if st.button('Batchfreigabe Nemetris'):
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        data_A.to_xml(f'Data_A_{now}.xml')
        data_B.to_xml(f'Data_B_{now}.xml')
        data_A.to_json(config['GA']['savepath']+'_A')
        data_B.to_json(config['GA']['savepath']+'_B')
        # os.copy(here, there)

# Main Functionality
if linie == 'A':
    data = data_A
elif linie == 'B':
    data = data_B

# Werkerlast Plot
lastplot_col1, lastplot_col2 = tab_chart.columns([0.5, 0.5])
op_cols = ['OP11', 'OP12', 'OP13', 'OP14', 'OP15', 'OP31', 'OP32', 'OP33', 'OP34', 'OP35', 'EOLMech', 'EOL_elektr.']
lastplot_containers = []
for idx, op in enumerate(op_cols):
    if idx % 2 == 0:
        col = lastplot_col1
    else:
        col = lastplot_col2
    container = col.container()
    lastplot_containers.append(container)
    data_op = data[op].values
    deadlines = data['Deadline']
    plot_werkerlast_bar(data_op, deadlines, op, container)
    
# Dataframe mit Daten
plot_frame = copy.deepcopy(data[['produktionsnummer', 'Deadline', 'auidnr']])
op_cols = ['OP11', 'OP12', 'OP13', 'OP14', 'OP15', 'OP31', 'OP32', 'OP33', 'OP34', 'OP35', 'EOLMech', 'EOL_elektr.']
plot_frame['MTM-Zeit'] = [[row[op_col] for op_col in op_cols] for idx, row in data.iterrows()]

tab_data.data_editor(
    plot_frame,
    column_config={
        "Deadline": st.column_config.TextColumn('Deadline', disabled=True),
        "MTM-Zeit": st.column_config.BarChartColumn(
            "Aufwand Pro Arbeitsplatz", y_min=0, y_max=250, width=400
        ),
    },
    hide_index=False,
    num_rows='fixed',
    height=2000
)

# Optimierungsdaten
opti_data = pd.read_csv("algorithm_statistics.csv", sep=",")
opti_data.columns = ["mean_workload_diff", "mean_tardiness", "min_workload_diff", "min_tardiness", "std_workload_diff", "std_tardiness"]
opti_col1, opti_col2 = tab_optimization.columns([0.5, 0.5])
plot_opti_data(opti_data, opti_col1)
