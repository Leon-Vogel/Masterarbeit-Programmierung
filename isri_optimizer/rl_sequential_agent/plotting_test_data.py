import pandas as pd
import numpy as np

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
    table += f"\\textbf{{Experiment}} & \\textbf{{Total Reward}} & \\textbf{{Deadline Gap}} & \\textbf{{Workload Gap}} & \\textbf{{Deadline Reward}} & \\textbf{{Diffsum Reward}} \\\\\n"
    table += "\\hline\n"

    for cluster_size in [8, 12, 15]:
        table += f"\\multicolumn{{6}}{{l}}{{\\textbf{{{cluster_size}}}}} \\\\\n"
        for i, cluster_method in enumerate(['kmeans', 'neighbour', 'no_cluster']):
            env_name = f"3_{experiment_type}_{cluster_size}_{cluster_method}"
            subset = filtered_df[filtered_df['env'] == env_name]
            if not subset.empty:
                row = subset.iloc[0]
                table += f"\\hspace{{1em}}{Cluster[i]} & {row['total_reward']:.2f} & {row['mean_deadline_gap']:.2f} & {row['mean_workload_gap']:.2f} & {row['mean_deadline_reward']:.2f} & {row['mean_diffsum_reward']:.2f} \\\\\n"
        table += "\\hline\n"

    table += "\\end{tabular}\n\\caption{Durchschnitt der Kennzahlen aus dem Test für " + Rewards[experiment_type] + "}\n\\end{table}\n"
    return table

latex_dense = create_latex_table(grouped_df, 'dense')
latex_sparse = create_latex_table(grouped_df, 'sparse')
latex_sparse_sum = create_latex_table(grouped_df, 'sparse_sum')

with open(f'{MODEL_SAVE_DIR}/results_tables.tex', 'w', encoding='utf-8') as f:
    f.write(latex_dense)
    f.write("\n\n")
    f.write(latex_sparse)
    f.write("\n\n")
    f.write(latex_sparse_sum)