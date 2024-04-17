import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.abspath(''), ".."))
from isri_optimizer.isri_on_premise import isri_optimize
from misc_utils.color_palette import colors
import streamlit as st

st.set_page_config(layout="wide")
# Durch DB Abfrage ersetzen 
data_A = pd.read_json('PlanA.json')
data_B = pd.read_json('PlanB.json')

# Plot Funktionen
@st.cache_resource
def plot_werkerlast_bar(data, border=120):
    fig, axs = plt.subplots(6, 2, figsize=[15, 24])
    axs = axs.flatten()
    names = ['OP11', 'OP12', 'OP13', 'OP14', 'OP15', 'OP31', 'OP32', 'OP33', 'OP34', 'OP35', 'EOL Mech', 'EOL_elektr.']
    xs = np.arange(0, len(data), 1)
    cmap = plt.get_cmap('RdYlGn')
    for idx, op in enumerate(names):
        heights = data[op].to_list()
        #cs = [colors['green']['shade_2'] if height <= border else colors['red']['shade_3'] for height in heights]
        cs = cmap(data['Deadline']/ data['Deadline'].max())
        axs[idx].bar(xs, heights, color=cs, alpha=0.8)
        axs[idx].set_title(op)
    
    return fig

# Control Panel
with st.sidebar:
    linie = st.selectbox('Linie: ', ['A', 'B'])
    goal = st.selectbox('Grafik: ', ['Werkerlast'])
    if st.button('Start'):
        config = 'config.ini'
        os.system("python isri_on_premise.py")
        st.write('Successful')

# Main Functionality
if linie == 'A':
    data = data_A
elif linie == 'B':
    data = data_B

if goal == 'Werkerlast':
    # plot_werkerlast_bar(data)
    st.pyplot(plot_werkerlast_bar(data), use_container_width=True)