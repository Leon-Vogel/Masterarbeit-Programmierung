from _dir_init import *
import random
import torch
import numpy as np
from misc_utils import gantt_plotter_matplotlib
from misc_utils import schedule_to_graph_converter


def generate_random_schedule(num_machines, num_jobs, num_ops_per_job):
    operations = []
    for job in range(num_jobs):
        n_ops = random.randint(num_ops_per_job[0], num_ops_per_job[1])
        for op in range(n_ops):
            op_time = random.randint(1, 10)
            op_machine = random.randint(1, num_machines)
            operations.append({"job": job, "op": op, "time": op_time, "machine": op_machine})
    operations.sort(key=lambda x: x["time"])
    schedule = [
        {
            "start_time": 0,
            "end_time": operations[0]["time"],
            "resource": operations[0]["machine"],
            "job": operations[0]["job"],
            "op": operations[0]["op"],
        }
    ]
    for op in operations[1:]:
        resource_predecessors_end_times = [s["end_time"] for s in schedule if s["resource"] == op["machine"]]
        job_predecessors_end_times = [s["end_time"] for s in schedule if s["job"] == op["job"]]
        earliest_start_by_pre_res = max(resource_predecessors_end_times) if resource_predecessors_end_times else 0
        earliest_start_by_pre_job = max(job_predecessors_end_times) if job_predecessors_end_times else 0
        op_start_time = max(earliest_start_by_pre_res, earliest_start_by_pre_job)
        schedule.append(
            {
                "start_time": op_start_time,
                "end_time": op_start_time + op["time"],
                "resource": op["machine"],
                "job": op["job"],
                "op": op["op"],
            }
        )
    return schedule

def generate_training_data(n_samples=200, n_jobs=(5, 20), n_operations=(1, 5), n_resources=(2, 5)):
    dat = []
    for _ in range(n_samples):
        N = random.randint(n_jobs[0], n_jobs[1])
        M = random.randint(n_resources[0], n_resources[1])
        dat.append(generate_random_schedule(M, N, n_operations))
    return dat


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv as CONV
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx
import random
import networkx as nx
from torch_geometric.data import Data

class GraphSageModel(nn.Module):
    def __init__(self, hidden_channels, out_channels, in_channels, dropout, learn_rate):
        super(GraphSageModel, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(CONV(in_channels, hidden_channels[0]))

        for i in range(1, len(hidden_channels)):
            self.convs.append(CONV(hidden_channels[i - 1], hidden_channels[i]))

        self.fc = nn.Linear(hidden_channels[-1], out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def data_loader(self, batch_size):
        def get_node_features(graph):
            node_features = []
            for node in graph.nodes():
                out_edges = graph.out_edges(node, data=True)
                if len(out_edges) == 0:
                    node_feature = torch.tensor([0, 0], dtype=torch.float)  # Platzhalter, falls es keine ausgehenden Kanten gibt
                else:
                    time_dists = [edge_data['time_dist'] for _, _, edge_data in out_edges]
                    node_feature = torch.tensor([time_dists[0], time_dists[1]], dtype=torch.float)
                node_features.append(node_feature)

            x = torch.stack(node_features)
            return x

        data_list = []
        for sample in samples:
            graph = sample['graph']
            x = get_node_features(graph)
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            edge_attr = torch.tensor([list(graph[u][v].values()) for u, v in graph.edges]).float()
            
            
            y = get_output(sample['schedule_data'])
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        return DataLoader(data_list, batch_size=batch_size, shuffle=True)

    def fit(self, device, samples, n_epochs, batch_size):

        data = [from_networkx(s["graph"], group_edge_attrs=s['edge_attr']) for s in samples]
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        for epoch in range(n_epochs):
            self.train()
            total_loss = 0
            for batch in data_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = self.loss_criterion(out, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            print("Epoch", epoch, "Loss", total_loss / len(data))


def train(samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSageModel(
        in_channels=2, hidden_channels=[256, 128, 64], out_channels=20, dropout=0.3, learn_rate=0.01
    ).to(device)
    model.fit(device, samples, n_epochs=100, batch_size=64)




def get_edge_features(graph):
    edge_list = list(graph.edges(data=True))
    num_nodes = len(list(graph.nodes()))

    time_dists = []
    edge_types = []
    connections = []
    for src, dst, attr in edge_list:
        td = attr['time_dist']
        et = attr['edge_type']
        time_dists.append(td)
        edge_types.append(et)
        connections.append((src,dst))

    # Knotenmerkmale sind die Konkatenation von eingehenden und ausgehenden Kantenmerkmalen
    node_features = torch.cat([in_edges, out_edges], dim=1)
    return node_features

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    # samples = generate_training_data()
    # gantt_plotter_matplotlib.plot_gantt(samples[0])
    samples = []
    for s in generate_training_data():
        graph, edge_attr = schedule_to_graph_converter.schedule_data_to_graph(s)
        samples.append({'schedule_data': s, 'graph': graph, 'edge_attr': edge_attr})

    #get_edge_features(samples[0]['graph'])

    train(samples)

    pass
