from _dir_init import *
import torch
import torch.nn as nn
import numpy as np
from multiprocessing import Pool
from copy import deepcopy
import random
from collections import deque
from misc_utils.copy_helper import fast_deepcopy
from data_preprocessing import IsriDataset
import pickle
from scipy.spatial.distance import pdist, squareform
from tqdm import trange

class ReplayBuffer:
    def __init__(self, max_size=1000) -> None:
        self.content = deque(maxlen=max_size)

    def add(self, experience):
        # experience (St-n, vt-n, Rt-n_sum, St, node_attrs, adjacancy)
        self.content.append(experience)
    
    @property
    def size(self):
        return len(self.content)
    
    def sample_dataset(self, n_samples=100):
        assert n_samples <= self.size, f"Current Size is {self.size} but {n_samples} requested"
        samples = np.random.choice(np.arange(0, self.size, 1), size=n_samples)
        samples = [self.content[s] for s in samples]
        return samples
    

class TSPModel(nn.Module):
    """Implementation of the TSP Version of https://arxiv.org/abs/1704.01665
    Learning Combinatorial Optimization Algorithms over Graphs - Dai et al."""
    def __init__(self, hidden_dim: int = 128, num_iters_s2v: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_iters_s2v = num_iters_s2v
        
        self.theta1 = nn.Parameter(data=torch.rand(hidden_dim), requires_grad=True)
        self.theta2 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)), requires_grad=True)
        self.theta3 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)), requires_grad=True)
        self.theta4 = nn.Parameter(data=torch.rand(hidden_dim), requires_grad=True)
        self.theta5 = nn.Parameter(data=torch.rand(2 * hidden_dim), requires_grad=True)
        self.theta6 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)), requires_grad=True)
        self.theta7 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)), requires_grad=True)
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()

    def struct_to_vec(self, node_attrs: torch.Tensor, edge_weights: torch.Tensor, num_iters: int = 2) -> torch.Tensor:
        """Implements the Graph convolution. The TSP Graph is a fully connected Graph - message passing
        happens between all nodes, no Neighbor calculation needed"""
        n_nodes = node_attrs.shape[0]
        mu_t = torch.zeros((n_nodes, self.hidden_dim))
        # Formula Page 4 Chapter 3.2
        # theta1 * x_v
        for iter in range(num_iters):
            node_attr_embedding = node_attrs.unsqueeze(1) * self.theta1.unsqueeze(0)

            # theta2 * sum over neighbors mu_t - Graph is fully connected so all nodes are neighbors and have the same neighborhood
            neighbor_embedding = self.theta2.mm(torch.sum(mu_t, dim=0).unsqueeze(1)).expand((self.hidden_dim, n_nodes)).permute(1, 0)

            # Theta3 * sum over neighbors u relu(theta4 * weight[u, v]) for each v
            weights = self.relu(self.theta4 * edge_weights.unsqueeze(2))
            weights_sum = torch.sum(weights, dim=1).permute(1, 0) # dim = 0?!
            weights_sum = self.theta3.mm(weights_sum).permute(1, 0)
            
            mu_t = self.relu(node_attr_embedding + neighbor_embedding + weights_sum)
        
        return mu_t
    
    def q_function(self, node_attrs, edge_weights):
        """Prediction of q values for each node Q(h(s), v; theta) im Paper aber hier direkt für alle v"""
        mu_T = self.struct_to_vec(node_attrs, edge_weights, self.num_iters_s2v)
        
        node_embeddings = self.theta7.mm(mu_T.T).T
        graph_embedding = self.theta6.mm(torch.sum(mu_T, dim=0).unsqueeze(1)).expand(self.hidden_dim, edge_weights.shape[0]).permute(1, 0) # repeat?

        embedding = torch.cat((node_embeddings, graph_embedding), dim=1)
        q = self.relu(embedding).mm(self.theta5.unsqueeze(1))
        return q.squeeze()

    def train_model_sequential(self, dataset: list, epochs: int = 100, n_step_q: int = 4, lr: float = 0.0001, gamma: float = 0.99) -> None:
        """Dataset should be a list of tuples with (points, adjacency)"""
        replay_buffer = ReplayBuffer()
        epsilon = 1
        epsilon_stepsize = 0.95 / epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        pbar = trange(epochs, desc='Training Epoch 1', leave=True)
        for epoch in pbar:
            random_problem = random.choice(list(range(len(dataset))))
            points, adjacency = dataset[random_problem]
            curr_state = []
            curr_tour_length = 0
            states_hist = deque(maxlen=n_step_q)
            vertex_hist = deque(maxlen=n_step_q)
            reward_hist = deque(maxlen=n_step_q)
            for step in range(points.shape[0]):
                next_v = self.select_vertex(points, adjacency, curr_state, epsilon=epsilon)
                states_hist.append(curr_state)
                vertex_hist.append(next_v)
                curr_state.append(next_v) # Möglicherweise durch Helferheuristik ersetzen
                reward, curr_tour_length  = self.get_reward(curr_tour_length, curr_state, adjacency)
                reward_hist.append(reward)
                if step > n_step_q:
                    replay_buffer.add([fast_deepcopy(states_hist[0]), vertex_hist[0], sum(reward_hist),
                                       fast_deepcopy(curr_state), points, adjacency])
                    loss = self.train_on_batch(replay_buffer, optimizer, gamma)
            epsilon -= epsilon_stepsize
            pbar.set_description("Training Epoch %e" % epoch)
            pbar.set_postfix({'Loss: ': loss})

    def select_vertex(self, points: torch.Tensor, adjacency: torch.Tensor, state: list, epsilon: float) -> int:
        epsilon_p = random.uniform(0, 1)
        if epsilon_p < epsilon:
            # Random Node
            v = random.choice([v for v in range(points.shape[0]) if v not in state])
            return v
        
        else:
            mask = torch.tensor([1 if v in state else 0 for v in range(points.shape[0])], dtype=torch.bool)
            node_attrs = ~mask # Hier könnte man auch Knoten Attribute einfügen
            q_values = self.q_function(node_attrs=node_attrs, edge_weights=adjacency)
            mask = [1 if v not in state else 0 for v in range(adjacency.shape[0])]
            q_values[mask] = -torch.inf
            argmax = torch.argmax(q_values)
            return argmax

    def train_on_batch(self, replay_buffer: ReplayBuffer, optimizer, gamma, batch_size: int = 64, epochs: int = 10):
        for epoch in range(epochs): 
            batch_size = min(batch_size, replay_buffer.size)
            train_batch = replay_buffer.sample_dataset(batch_size)
            predictions = []
            targets = []
            # TODO: Parallelisieren
            optimizer.zero_grad()
            for sample in train_batch:
                # sample: (St-n, vt-n, Rt-n_sum, St, node_attrs, adjacancy)
                node_attrs = torch.tensor([0 if v in sample[3] else 1 for v in range(sample[4].shape[0])], dtype=torch.bool)
                curr_value = self.q_function(node_attrs, sample[5])
                curr_value = torch.argmax(curr_value).item()
                target_y = sample[2] + curr_value
                node_attrs = torch.tensor([0 if v in sample[3] else 1 for v in range(sample[5].shape[0])], dtype=torch.bool)
                prediction = self.q_function(node_attrs, sample[5])[sample[1]]
                targets.append(target_y)
                predictions.append(prediction)
            
            
            loss = self.loss(torch.stack(predictions), torch.tensor(targets))
            loss.backward()
            optimizer.step()
        return loss.item()

    def get_reward(self, reward_old, state_new: list, adjacency: torch.Tensor):
        total_distance = sum([adjacency[state_new[idx], state_new[idx + 1]].item() for idx in range(len(state_new) - 1)])
        difference = reward_old - total_distance
        return difference, total_distance

def create_dataset(n_samples: int, isri_dataset_path: str) -> list:
    dataset = pickle.load(open(isri_dataset_path, 'rb'))
    sample_idxs = random.sample(list(range(dataset.data_size)), k=n_samples)
    data = []
    for sample in sample_idxs:
        jobdata = dataset.data['Jobdata'][sample]
        points = np.array([jobdata[job]['times'] for job in jobdata.keys()]) / 1000
        adjacency = torch.tensor(squareform(pdist(points)))
        data.append([torch.tensor(points, dtype=torch.float32), adjacency.to(torch.float32)])
    return data

if __name__ == '__main__':
    isri_dataset_path = 'isri_optimizer/rl_sequential_agent/IsriDataset.pkl'
    n_samples = 100
    dataset = create_dataset(n_samples=n_samples, isri_dataset_path=isri_dataset_path)
    model = TSPModel()
    model.train_model_sequential(dataset)
