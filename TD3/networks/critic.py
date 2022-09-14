import torch as torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from config import LEARNING_RATE


class Critic(nn.Module):

    def __init__(self, states_dimension, actions_dimension):
        super(Critic, self).__init__()
        self.checkpoint_path = "critic"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Red para el crítico 1
        self.fc_1 = nn.Linear(states_dimension + actions_dimension, 400)
        self.fc_2 = nn.Linear(400, 300)
        self.fc_3 = nn.Linear(300, 1)

        # Red para el crítico 2
        self.fc_4 = nn.Linear(states_dimension + actions_dimension, 400)
        self.fc_5 = nn.Linear(400, 300)
        self.fc_6 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.to(self.device)

    def forward(self, state, action):
        _x = torch.cat([state, action], 1)

        # Forward para el crítico 1
        _x_1 = functional.relu(self.fc_1(_x))
        _x_1 = functional.relu(self.fc_2(_x_1))
        _x_1 = self.fc_3(_x_1)

        # Forward para el crítico 2
        _x_2 = functional.relu(self.fc_4(_x))
        _x_2 = functional.relu(self.fc_5(_x_2))
        _x_2 = self.fc_6(_x_2)

        return _x_1, _x_2
  
    def forward_Q_value_1(self, state, action):
        # Forward solo para el primer crítico
        _x = torch.cat([state, action], 1)

        _x_1 = functional.relu(self.fc_1(_x))
        _x_1 = functional.relu(self.fc_2(_x_1))
        _x_1 = self.fc_3(_x_1)

        return _x_1

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_path))
