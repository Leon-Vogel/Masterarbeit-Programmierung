import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

CUPS_PER_WEEK = 8 * 60
BUFFER = 2.2
DATAPATH = "./data"


def create_setupmatrix(numberOfTypes):
    Path(DATAPATH).mkdir(parents=True, exist_ok=True)
    x_coord = np.random.randint(low=1, high=100, size=(numberOfTypes, 1))
    y_coord = np.random.randint(low=1, high=100, size=(numberOfTypes, 1))

    set_up_matrix = np.zeros((numberOfTypes, numberOfTypes))

    for i in range(0, numberOfTypes):
        for j in range(0, numberOfTypes):
            if x_coord[i] - x_coord[j] >= 0:
                alpha = 1.5
            else:
                alpha = 1

            if y_coord[i] - y_coord[j] >= 0:
                beta = 1.5
            else:
                beta = 1

            set_up_matrix[i, j] = alpha * abs(x_coord[i] - x_coord[j]) + beta * abs(y_coord[i] - y_coord[j])

    print(set_up_matrix)

    # set_up_matrix = np.random.randint(low=1, high=24, size=(numberOfTypes, numberOfTypes))
    # np.fill_diagonal(set_up_matrix, 0)
    # set_up_matrix = set_up_matrix*10

    pd.DataFrame(set_up_matrix).to_csv(os.path.join(DATAPATH, f"setupmatrix_{numberOfTypes}.csv"))
    np.save(os.path.join(DATAPATH, f"setupmatrix_{numberOfTypes}.npy"), set_up_matrix)
    # with open(os.path.join(DATAPATH, "setupmatrix.pkl"), 'wb') as f:
    #     pickle.dump(set_up_matrix, f)


def create_random_instances(numberOfOrders, numberOfTypes, instanceName):
    Path(DATAPATH).mkdir(parents=True, exist_ok=True)
    random_instances = pd.DataFrame(columns=['ID', 'type', 'number_of_cups'])
    for i in range(numberOfOrders):
        local_type = np.random.randint(low=0, high=numberOfTypes)
        local_lot_size = np.random.randint(low=1, high=20) * 100
        random_instances.loc[i] = [i, local_type, local_lot_size]

    final_week = np.ceil(random_instances['number_of_cups'].sum() / CUPS_PER_WEEK)
    final_week *= (1 + BUFFER)

    for i in range(numberOfOrders):
        random_instances.at[i, 'deadline'] = np.random.randint(low=1, high=final_week)

    random_instances["processing_time"] = random_instances["number_of_cups"] / CUPS_PER_WEEK

    random_instances.to_csv(os.path.join(DATAPATH, f"{instanceName}.csv"))
    random_instances.to_pickle(os.path.join(DATAPATH, f"{instanceName}.pkl"))


if __name__ == "__main__":
    create_setupmatrix(4)
    create_random_instances(10, 4, "10-4")