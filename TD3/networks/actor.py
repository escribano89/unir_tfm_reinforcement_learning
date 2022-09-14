import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from config import LEARNING_RATE


class Actor(nn.Module):
    def __init__(self, states_dimension, actions_dimension, max_action):
        super(Actor, self).__init__()

        self.checkpoint_path = "actor"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = max_action
    
        self.fc_1 = nn.Linear(states_dimension, 400)
        self.fc_2 = nn.Linear(400, 300)
        self.fc_3 = nn.Linear(300, actions_dimension)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.to(self.device)

    def forward(self, x):
        _x = functional.relu(self.fc_1(x))
        _x = functional.relu(self.fc_2(_x))
        _x = self.max_action * torch.tanh(self.fc_3(_x))

        return _x

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_path))
