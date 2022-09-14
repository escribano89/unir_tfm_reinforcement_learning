import torch as torch
import torch.nn as nn
import torch.optim as optim
from config import LEARNING_RATE


class Critic(nn.Module):
    def __init__(self, input_dimensions):
        super(Critic, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "critic"

        self.fc_1 = nn.Linear(*input_dimensions, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 64)
        self.v = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(self.device)

    def forward(self, state):
        _x = torch.tanh(self.fc_1(state))
        _x = torch.tanh(self.fc_2(_x))
        _x = torch.tanh(self.fc_3(_x))

        return self.v(_x)

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_path))
