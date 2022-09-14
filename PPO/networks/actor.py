import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from config import LEARNING_RATE
from torch.distributions import Beta


class Actor(nn.Module):
    def __init__(self, number_of_actions, input_dimensions):
        super(Actor, self).__init__()
        self.checkpoint_path = "actor"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta_constant = 1.0

        # fc -> Fully connected layer
        self.fc_1 = nn.Linear(*input_dimensions, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 64)

        # Distribución Beta - Revisiting Design Choices in Proximal Policy Optimization
        # https://arxiv.org/abs/2009.10897
        # The authors chose Beta policy parameterizations because they can explicitly incorporate action space
        # boundaries and eliminate the biased boundary effects caused by truncated Gaussian
        # It also leads to more reliable convergence behavior of the PPO algorithm in our test cases and can
        # outperform standard PPO even in settings where boundary effects are not relevant.
        self.alpha = nn.Linear(64, number_of_actions)
        self.beta = nn.Linear(64, number_of_actions)

        # Adam Optimizer -> https://arxiv.org/abs/1412.6980
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.to(self.device)

    def forward(self, state):
        _x = torch.tanh(self.fc_1(state))
        _x = torch.tanh(self.fc_2(_x))
        _x = torch.tanh(self.fc_3(_x))

        # Distribución beta - Necesita ser positivo. Asegurar que no sea 0
        # o un número muy negativo
        _alpha = functional.relu(self.alpha(_x)) + self.beta_constant
        _beta = functional.relu(self.beta(_x)) + self.beta_constant
 
        return Beta(_alpha, _beta)

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_path))
