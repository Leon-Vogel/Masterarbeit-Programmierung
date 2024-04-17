import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

DATAPATH = "./data"
CUPS_PER_WEEK = 8 * 60


def integrate_setimes(schedule, se_matrix):
    rows_to_insert = []
    for i in range(1, len(schedule)):
        if schedule.at[i, 'type'] != schedule.at[i - 1, 'type']:
            setup_effort = se_matrix[schedule.at[i - 1, 'type'], schedule.at[i, 'type']]
            rows_to_insert.append((i, {'ID': -1, 'type': -1, 'number_of_cups': setup_effort,
                                       'deadline': -1, 'processing_time': setup_effort/CUPS_PER_WEEK}))
    for index, row in reversed(rows_to_insert):
        schedule = pd.concat(
            [schedule.iloc[:index], pd.DataFrame([row], columns=schedule.columns), schedule.iloc[index:]]).reset_index(
            drop=True)
    return schedule

def read_data(datapath, number_of_orders, number_of_types):
    # with open(os.path.join(DATAPATH, "setupmatrix.pkl"), 'rb') as f:
    #     self.se_matrix = pickle.load(f)
    se_matrix = np.load(os.path.join(datapath, f"setupmatrix_{number_of_types}.npy"))
    # with open(os.path.join(datapath, "20-8.pkl"), 'rb') as f:
    #     schedule = pickle.load(f)
    schedule = pd.read_pickle(os.path.join(datapath, f"{number_of_orders}-{number_of_types}.pkl"))
    print(se_matrix)
    print(schedule)

    schedule_se = integrate_setimes(schedule, se_matrix)

    schedule_se['start'] = schedule_se["processing_time"].shift(fill_value=0).cumsum()
    schedule_se['end'] = schedule_se["processing_time"].cumsum()
    print(schedule_se)
    return schedule, se_matrix, schedule_se

def evaluate_schedule(schedule_se: pd.DataFrame) -> dict:
    schedule = schedule_se[schedule_se["deadline"] != -1]
    se_sum = schedule_se.loc[schedule_se["deadline"] == -1, "processing_time"].sum()
    schedule["tardiness"] = schedule["deadline"] - schedule["end"]
    bool_dd_violated = schedule["tardiness"] < 0
    metrics = {
        "makespan": schedule_se["end"].iloc[-1],
        "tardiness_sum": schedule[bool_dd_violated]["tardiness"].abs().sum(),
        "se_sum": se_sum,
    }
    return metrics




class ShiftCurrentJobAction:
    """1. action im multidiskreten raum: soll der aktuelle job
    in der produktionsreihenfolge verschoben werden?"""

    DO_NOTHING = 0
    SHIFT_RIGHT = 1
    SHIFT_LEFT = 2


class SelectNextJobAction:
    """2. action im multidiskreten raum: soll der job für die nächste action
    beibehalten werden oder soll ein neuer zu shiftender job
    gem. einer heuristik ausgewählt werden?"""

    DO_NOTHING = 0
    GET_JOB_WITH_HIGH_FLOWTIME = 1
    GET_JOB_WITH_LOW_FLOWTIME = 2
    GET_JOB_WITH_HIGH_SLACKMEAN = 3
    GET_JOB_WITH_LOW_SLACKMEAN = 4
    GET_JOB_WITH_HIGH_DUETIME = 5
    GET_JOB_WITH_LOW_DUETIME = 6
    GET_JOB_WITH_HIGHEST_DUETIME = 7
    GET_JOB_WITH_LOWEST_DUETIME = 8


def visualize(schedule, color_col="type"):
    fig = go.Figure()

    for index, row in schedule.iterrows():
        fig.add_shape(type="rect",
                      x0=row['start'], x1=row['end'], y0=0, y1=1,
                      fillcolor=px.colors.qualitative.Plotly[int(row[color_col])], #  % len(px.colors.qualitative.Plotly)
                      line=dict(width=0))

    fig.update_layout(showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                                 range=[0, schedule['end'].max()]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor='rgba(0,0,0,0)')

    fig.show()

if __name__ == "__main__":
    schedule, se_matrix, schedule_se = read_data(DATAPATH, number_of_orders=10, number_of_types=4)
    # visualize(schedule_se)
    print(evaluate_schedule(schedule_se))
