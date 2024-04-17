import graph_prediction
import schedule_to_graph_converter

X = []
for s in graph_prediction.generate_training_data():
    graph, edge_attr = schedule_to_graph_converter.schedule_data_to_graph(s)
    X.append(graph)

import numpy as np
import torch
def add_node_features(graph):
    for node in graph.nodes():
        out_edges = graph.out_edges(node, data=True)
        n_outs = len(out_edges)
        if n_outs == 0:
            node_feature = np.array([0, 0, 0, 0])
        else:
            time_dists = [edge_data['time_dist'] for _, _, edge_data in out_edges]
            edge_types = [edge_data['edge_type'] for _, _, edge_data in out_edges]
            if n_outs == 1:
                node_feature = np.array([time_dists[0], 0, edge_types[0], 0])
            elif n_outs == 2:
                node_feature = np.array([time_dists[0], time_dists[1], edge_types[0], edge_types[1]])
        graph.nodes[node]['features'] = node_feature
    return graph

X = [add_node_features(g) for g in X]



import networkx as nx
from torch_geometric.utils import from_networkx

def convert_to_torch_geometric(graph: nx.DiGraph):
    data = from_networkx(graph, group_node_attrs=['features'], group_edge_attrs=edge_attr)
    features = torch.tensor([graph.nodes[node]['features'] for node in graph.nodes()], dtype=torch.float)
    data.x = features
    return data


torch_geometric_graphs = [convert_to_torch_geometric(g) for g in X]


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, num_layers):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_features, hidden_features))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_features, hidden_features))
        self.convs.append(SAGEConv(hidden_features, output_features))

    def forward(self, x, edge_index):
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

input_features = 4  # Anzahl der Eingabemerkmale (Knotenattribute)
hidden_features = 64  # Anzahl der versteckten Merkmale
output_features = 20  # Anzahl der gewünschten Ausgabemerkmale
num_layers = 3  # Anzahl der Schichten in Ihrem GraphSAGE-Modell

model = GraphSAGEModel(input_features, hidden_features, output_features, num_layers)



from torch_geometric.data import DataLoader, Batch

# Datensatz erstellen
dataset = DataLoader(torch_geometric_graphs, batch_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Trainingsfunktion
def train(epoch):
    model.train()
    total_loss = 0
    for batch in dataset:
        optimizer.zero_grad()
        x, edge_index = batch.x, batch.edge_index
        out = model(x, edge_index)
        loss = F.mse_loss(out, x)  # Verlustfunktion (Mean Squared Error)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)

# Training
num_epochs = 100
for epoch in range(num_epochs):
    loss = train(epoch)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}')