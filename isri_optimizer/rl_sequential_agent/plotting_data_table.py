import pandas as pd
import numpy as np
from plotting_funktion import read_tensorflow_events, find_event_files_dict

MODEL_SAVE_DIR = "./isri_optimizer/rl_sequential_agent/savefiles_Train1/"

df = pd.read_csv(f'{MODEL_SAVE_DIR}/testing_results.csv',)
df = df.drop(columns=['test_run', 'total_steps'])
grouped_df = df.groupby('env').mean().reset_index()
Rewards = {
    'dense': 'häufige Rewards',
    'sparse': 'seltene Rewards, relativ zum genetischen Algorithmus',
    'sparse_sum': 'seltene Rewards'
}
Cluster = ['Kmeans', 'Agglomerativ', 'Ohne Cluster']
def create_latex_table(df, experiment_type):
    filtered_df = df[df['env'].str.contains(experiment_type)]
    table = (
        "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lccccc}\n"
        + "\\hline\n"
    )
    table += f"\\textbf{{Experiment}} & \\textbf{{Return}} & \\textbf{{D. Gap}} & \\textbf{{W. Gap}} & \\textbf{{D. Return}} & \\textbf{{W. Return}} \\\\\n"
    table += "\\hline\n"

    for cluster_size in [8, 12, 15]:
        Aktionen = f'{cluster_size} Aktionen'
        table += f"\\multicolumn{{6}}{{l}}{{\\textbf{{{Aktionen}}}}} \\\\\n"
        for i, cluster_method in enumerate(['kmeans', 'neighbour', 'no_cluster']):
            env_name = f"3_{experiment_type}_{cluster_size}_{cluster_method}"
            subset = filtered_df[filtered_df['env'] == env_name]
            if not subset.empty:
                row = subset.iloc[0]
                table += f"\\hspace{{1em}}{Cluster[i]} & {row['total_reward']:.3f} & {row['mean_deadline_gap']:.3f} & {row['mean_workload_gap']:.3f} & {row['mean_deadline_reward']:.3f} & {row['mean_diffsum_reward']:.3f} \\\\\n"
        table += "\\hline\n"

    table += "\\end{tabular}\\vspace{{0.4cm}}\n\\caption{Durchschnitt der Kennzahlen aus dem Test für " + Rewards[experiment_type] + "}\n\\end{table}\n"
    return table

def create_latex_table_from_events(experiments, Rewards, Cluster):
    tables = {}
    
    for experiment_type, values in experiments.items():
        table = (
            "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lccccc}\n"
            + "\\hline\n"
        )
        table += f"\\textbf{{Experiment}} & \\textbf{{Return}} & \\textbf{{D. Gap}} & \\textbf{{W. Gap}} & \\textbf{{D. Return}} & \\textbf{{W. Return}} \\\\\n"
        table += "\\hline\n"

        for cluster_size, paths in values.items():
            Aktionen = f'{cluster_size.replace("_","")} Aktionen'
            table += f"\\multicolumn{{6}}{{l}}{{\\textbf{{{Aktionen}}}}} \\\\\n"
            data = read_tensorflow_events(experiments, experiment_type, cluster_size)
            for i, (cluster_method, path) in enumerate(zip(['kmeans', 'neighbour', 'no_cluster'], paths)):

                total_rewards = data['y_rew'][i][-50:]
                deadline_gaps = data['y_tard'][i][-50:]
                workload_gaps = data['y_diff'][i][-50:]
                deadline_rew = data['y_tard_rew'][i][-50:]
                workload_rew = data['y_diff_rew'][i][-50:]

                total_reward_mean = np.mean(total_rewards)
                deadline_gap_mean = np.mean(deadline_gaps)
                workload_gap_mean = np.mean(workload_gaps)
                deadline_rew_mean = np.mean(deadline_rew)
                workload_rew_mean = np.mean(workload_rew)

                table += (
                    f"\\hspace{{1em}}{Cluster[i]} & {total_reward_mean:.3f} & "
                    f"{deadline_gap_mean:.3f} & "
                    f"{workload_gap_mean:.3f}& "
                    f"{deadline_rew_mean:.3f} & "
                    f"{workload_rew_mean:.3f} \\\\\n"
                )
            table += "\\hline\n"

        table += (
            "\\end{tabular}\\vspace{{0.4cm}}\n\\caption{Durchschnitt der Kennzahlen aus den letzten 50 Trainingsiterationen für " 
            + Rewards[experiment_type] 
            + "}\n\\end{table}\n"
        )
        tables[experiment_type] = table
    
    return tables

test_table = True
if test_table:
    latex_dense = create_latex_table(grouped_df, 'dense')
    latex_sparse = create_latex_table(grouped_df, 'sparse')
    latex_sparse_sum = create_latex_table(grouped_df, 'sparse_sum')

    with open(f'{MODEL_SAVE_DIR}/results_test_tables.tex', 'w', encoding='utf-8') as f:
        f.write(latex_dense)
        f.write("\n\n")
        f.write(latex_sparse)
        f.write("\n\n")
        f.write(latex_sparse_sum)
train_table = True
if train_table:
    #base_directory = 'isri_optimizer/rl_sequential_agent/savefiles_Train1'
    event_file_paths = find_event_files_dict(MODEL_SAVE_DIR)

    Rewards = {
        'dense': 'häufige Rewards',
        'sparse': 'seltene Rewards, relativ zum genetischen Algorithmus',
        'sparse_sum': 'seltene Rewards'
    }
    Cluster = ['Kmeans', 'Agglomerativ', 'Ohne Cluster']

    latex_tables = create_latex_table_from_events(event_file_paths, Rewards, Cluster)

    with open(f'{MODEL_SAVE_DIR}/results_train_tables.tex', 'w', encoding='utf-8') as f:
        for table in latex_tables.values():
            f.write(table)
            f.write("\n\n")