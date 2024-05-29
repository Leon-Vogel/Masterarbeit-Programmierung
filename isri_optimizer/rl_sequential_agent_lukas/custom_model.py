from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from gymnasium import spaces
import torch
from torch import nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from misc_utils.igraph_helper import pad_array
import math
from stable_baselines3.common.torch_layers import MlpExtractor


def remove_padding(tensor, dim, padding_value):
    # Select the first element in each dimension except the given dimension
    # to create a vector of numbers.
    selection_indices = [0] * len(tensor.shape)
    selection_indices[dim] = slice(None)  # Take all elements in the given dimension
    vector = tensor[selection_indices]
    
    # Find the first occurrence of the padding value in the vector.
    if padding_value == 'nan':
        padding_idx = torch.where(torch.isnan(vector))[0]
    else:
        padding_idx = (vector == padding_value).nonzero(as_tuple=True)[0]
    
    # If padding_idx is empty, it means padding value was not found; 
    # use the entire dimension length
    if padding_idx.nelement() == 0:
        padding_idx = tensor.shape[dim]
    else:
        padding_idx = padding_idx[0].item()  # Convert to Python int
    
    # Slice the original tensor to exclude the padding values in the given dimension.
    slice_indices = [slice(None)] * len(tensor.shape)
    slice_indices[dim] = slice(None, padding_idx)
    
    result = tensor[slice_indices]
    
    return result


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class CustomDictTransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, input_features_dim: int = 13, n_heads: int=4, hidden_dim: int=64, num_layers: int=2):
        super(CustomDictTransformerFeatureExtractor, self).__init__(observation_space, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.input_embedding = nn.Linear(input_features_dim, hidden_dim)
        self.tgt_embedding = nn.Linear(input_features_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, batch_first=True, dim_feedforward=512)

    def forward(self, observations: dict) -> torch.Tensor:
        input_sequence = observations['input_sequence']
        tgt_sequence = observations['tgt_sequence']
        # tgt_sequence = remove_padding(tgt_sequence, dim=1, padding_value=-1)
        input_sequence = self.input_embedding(input_sequence)
        tgt_sequence = self.tgt_embedding(tgt_sequence)
        input_sequence_encoded = self.pos_encoder(input_sequence)
        tgt_sequence_encoded = self.pos_encoder(tgt_sequence)
        transformer_out = self.transformer(input_sequence_encoded, tgt_sequence_encoded)
        return transformer_out[:, -1, :]

    
class CustomLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, input_features_dim: int = 13, n_heads: int=4, hidden_dim: int=64, num_layers: int=2):
        super(CustomLSTMExtractor, self).__init__(observation_space, hidden_dim + 15)

        self.input_embedding = nn.Linear(input_features_dim, hidden_dim)
        self.tgt_embedding = nn.Linear(input_features_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observations: dict) -> torch.Tensor:
        input_sequence = observations['input_sequence']
        tgt_sequence = observations['tgt_sequence']
        # tgt_sequence = remove_padding(tgt_sequence, dim=1, padding_value=-1)
        input_sequence = self.input_embedding(input_sequence)
        tgt_sequence = self.tgt_embedding(tgt_sequence)
        input_embeddings, (h_t, c_t) = self.lstm(input_sequence)
        
        attention = tgt_sequence.matmul(h_t.permute(1, 2, 0))
        batch, next_n, hidden = tgt_sequence.shape
        if batch == 1:
            weighted_tgts = tgt_sequence + input_embeddings[:, -1]
        else:
            weighted_tgts = tgt_sequence + input_embeddings[:, -1].unsqueeze(1)
        value = attention.permute(0, 2, 1).matmul(weighted_tgts)

        if batch > 1:
            x = 6
        return torch.cat([value.squeeze(1), attention.squeeze(-1)], dim=1)


class CustomLSTMExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, input_features_dim: int = 13, n_heads: int=4, hidden_dim: int=64, num_layers: int=2):
        super(CustomLSTMExtractor2, self).__init__(observation_space, hidden_dim * 15)

        self.input_embedding = nn.Linear(input_features_dim, hidden_dim)
        self.tgt_embedding = nn.Linear(input_features_dim, hidden_dim)
        self.lstm_last_n = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lstm_next_n = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, observations: dict) -> torch.Tensor:
        input_sequence = observations['input_sequence']
        tgt_sequence = observations['tgt_sequence']
        
        # tgt_sequence = remove_padding(tgt_sequence, dim=1, padding_value=-1)
        input_sequence = self.input_embedding(input_sequence)
        tgt_sequence = self.tgt_embedding(tgt_sequence)
        batch, next_n, hidden = tgt_sequence.shape
        input_embeddings, (h_t_last, c_t_last) = self.lstm_last_n(input_sequence)
        # tgt_embeddings, (h_t_next, c_t_next) = self.lstm_next_n(tgt_sequence)
        # output = tgt_embeddings - h_t_last.permute(1, 0, 2)
        # values = tgt_sequence.matmul(input_embeddings.permute(0, 2, 1))
        flat_inputs = input_embeddings.flatten(1)
        attention_input = torch.cat([tgt_sequence, flat_inputs.unsqueeze(1).repeat(1, next_n, 1)], dim=2)
        output = self.attention(attention_input)
        return output.reshape(batch, next_n * hidden)
    
    
class TransformerPointingPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
    
