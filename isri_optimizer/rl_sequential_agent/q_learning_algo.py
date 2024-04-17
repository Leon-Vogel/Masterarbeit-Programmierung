from _dir_init import *
import torch
import torch.nn as nn
import numpy as np
from multiprocessing import Pool
from copy import deepcopy
import random


class ReplayBuffer:
    def __init__(self, max_size=1000) -> None:
        self.size = 0
        self.max_size = max_size
        self.content = []

    def add(self, experience):
        # experience (state_t, action_t, action_t_index, reward_sum, state_tn, done)
        self.content.append(experience)
        if self.size > self.max_size:
            self.content.pop(0)
        else:
            self.size += 1
    
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
        
        self.theta1 = nn.Parameter(data=torch.rand(hidden_dim))
        self.theta2 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)))
        self.theta3 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)))
        self.theta4 = nn.Parameter(data=torch.rand(hidden_dim))
        self.theta5 = nn.Parameter(data=torch.rand(2 * hidden_dim))
        self.theta6 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)))
        self.theta7 = nn.Parameter(data=torch.rand((hidden_dim, hidden_dim)))
        self.relu = nn.ReLU()

    def struct_to_vec(self, node_attrs: torch.Tensor, edge_weights: torch.Tensor, num_iters: int = 2) -> torch.Tensor:
        """Implements the Graph convolution. The TSP Graph is a fully connected Graph - message passing
        happens between all nodes, no Neighbor calculation needed"""

        mu_t = torch.zeros((node_attrs.shape[0], node_attrs.shape[0]))
        # Formula Page 4 Chapter 3.2
        # theta1 * x_v
        for iter in range(num_iters):
            node_attr_embedding = self.theta1.bmm(node_attrs)

            # theta2 * sum over neighbors mu_t - Graph is fully connected so all nodes are neighbors
            neighbor_messages = self.theta2.bmm(torch.sum(mu_t, dim=0))

            # Theta3 * sum over neighbors relu(theta4 * weight)
            neighbor_weights = self.relu(self.theta4.bmm(torch.sum(edge_weights, dim=1)))
            neighbor_weights = self.theta3.bmm(neighbor_weights)

            mu_t = self.relu(node_attr_embedding + neighbor_messages + neighbor_weights)
        
        return mu_t
    
    def q_function(self, node_attrs, edge_weights, mask):
        """Prediction of q values for each node Q(h(s), v; theta) im Paper aber hier direkt für alle v"""
        mu_T = self.struct_to_vec(node_attrs, edge_weights, self.num_iters_s2v)
        
        node_embeddings = self.theta7.bmm(mu_T)
        graph_embedding = self.theta6.bmm(torch.sum(mu_T, dim=0)) # repeat?

        embedding = torch.cat(node_embeddings, graph_embedding)
        q = self.theta5.T.bmm(self.relu(embedding))
        q[mask] = -torch.inf
        return q

    def train_model_sequential(self, dataset: list, epochs: int = 100) -> None:
        """Dataset should be a list of tuples with (points, adjacency)"""
        for epoch in range(epochs):
            random_problem = random.choice(list(range(len(dataset))))
            points, adjacency = dataset[random_problem]
            state = []
            node_attributes = torch.zeros(points.shape[0])
            for step in range(points.shape[0]):
                next_v = self.select_vertex(points, adjacency, state)
    
    def select_vertex(self, points: torch.Tensor, adjacency: torch.Tensor, state: list, epsilon: float) -> int:
        
        epsilon_p = random.uniform(0, 1)
        if epsilon_p < epsilon:
            # Random Node
            v = random.choice([v for v in range(points.shape[0]) if v not in state])
            return v
        
        else:
            mask = torch.tensor([0 if v in state else 1 for v in range(points.shape[0])], dtype=torch.bool)
            node_attrs = torch.flip(mask)
            q_values = self.q_function(node_attrs=node_attrs, edge_weights=adjacency, mask=mask)
            argmax = torch.argmax(q_values)
            return argmax

    def collect_experiences(agent_service, model, num_workers, n_experiences, buffer, replay_history_length=1):
        with Pool(num_workers) as pool:
            experiences_made = 0
            while experiences_made < n_experiences:
                results = pool.imap(play_episode, [(deepcopy(agent_service), deepcopy(model), replay_history_length)] * num_workers)
                for experience_list in results:
                    for experience in experience_list:
                        buffer.add(experience)
                        experiences_made += 1

    def get_reward(self, s_old, s_new):
        pass


def play_episode(input):
        agent_service = input[0]
        model = input[1]
        replay_history_length = input[2]
        observation_old = agent_service.get_new_start_ind_and_reset_env_and_create_initial_observation()
        done = False
        state_memory = []
        action_memory = []
        reward_memory = []
        action_index_memory = []
        step = 0
        experiences = []
        while step < agent_service.args['episode_len']:
            q_value, action, action_idx = model(observation_old, epsilon=0.05)
            state_memory.append(deepcopy(observation_old))
            action_index_memory.append(deepcopy(action_idx))
            observation_new, reward, done, info = agent_service.perform_step(action)
            action_memory.append(deepcopy(action))
            reward_memory.append(reward)
            
            if step >= replay_history_length:
                # Speichere Erfahrungen
                state_memory.pop(0)
                action_memory.pop(0)
                reward_memory.pop(0)
                action_index_memory.pop(0)
                reward_sum = np.sum(reward_memory)
                state_tn = state_memory[-1]
                experience = (state_memory[0], action[0], action_index_memory[0], reward_sum, state_tn, done)
                experiences.append(experience) 

            step += 1
            observation_old = observation_new
            if done:
                break
                
        return experiences
    

