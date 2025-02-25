# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def encode_state(state):
    """
    Encodes a state as a fixed-size feature vector.
    Uses two features:
      - The number of nodes in the state.
      - The total length of all nodes (as strings).
    """
    num_nodes = len(state.nodes)
    total_len = sum(len(str(node)) for node in state.nodes)
    return torch.tensor([num_nodes / 10.0, total_len / 100.0], dtype=torch.float32)

class AlphaZeroNet(nn.Module):
    """
    A neural network with a dual-head architecture:
      - The policy head outputs logits over a fixed maximum action space.
      - The value head outputs a scalar estimating the final reward.
    """
    def __init__(self, input_size, hidden_size, max_actions):
        super(AlphaZeroNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, max_actions)
        self.fc_value = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        policy_logits = self.fc_policy(h)  # Shape: (max_actions,)
        value = torch.tanh(self.fc_value(h))
        return policy_logits, value
