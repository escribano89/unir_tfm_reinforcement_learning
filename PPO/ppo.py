import torch as torch
import torch.nn as nn

from config import (BATCH_SIZE, DISCOUNT_FACTOR, ENTROPY_COEFFICIENT, EPOCHS,
                    GAE_LAMBDA, GRADIENT_CLIPPING, POLICY_CLIP)
from memory import Memory
from networks.actor import Actor
from networks.critic import Critic


class PPO:
    def __init__(self, number_of_actions, input_dimensions):
        self.memory = Memory(BATCH_SIZE)

        self.actor = Actor(number_of_actions, input_dimensions)
        self.actor.apply(self._orthogonal_initialization)

        self.critic = Critic(input_dimensions)
        self.critic.apply(self._orthogonal_initialization)

    # Inicialización ortogonal - Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks
    # https://arxiv.org/abs/2001.05992
    def _orthogonal_initialization(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def select_action(self, observation):
        # Deshabilita calculos de gradiente para inferencia y reduce consumo de memoria
        with torch.no_grad():
            _state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

            _distribution = self.actor(_state)
            _sampled_action = _distribution.sample()
            _probabilities = _distribution.log_prob(_sampled_action)

        # Devolvemos la acción sampleada y el probabilidad logaritmica asociada
        # en base a la distribución Beta
        # Ejecutamos flatten para eliminar una de las dimensiones adicionales
        # En el entorno de open AI gym no usamos tensores, sino arrays de numpy
        return _sampled_action.cpu().numpy().flatten(), _probabilities.cpu().numpy().flatten()

    # High-Dimensional Continuous Control Using Generalized Advantage Estimation
    # https://arxiv.org/abs/1506.02438

    # Resources:
    # https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
    # https://github.com/philtabor/Advanced-Actor-Critic-Methods/blob/main/PPO/single/continuous/agent.py
    def get_advantage_and_returns(self, memories):
        _states, _next_states, _rewards, _dones = memories

        # Deshabilita calculos de gradiente
        with torch.no_grad():
            _small_constant = 1e-4

            # Obtenemos los valores del crítico para
            # los estados actuales y los siguientes
            _values = self.critic(_states)
            _next_values = self.critic(_next_states)

            # Calculo de los deltas
            _deltas = _rewards + DISCOUNT_FACTOR * _next_values - _values
            _deltas = _deltas.cpu().flatten().numpy()

            _advantages = [0]
            for _delta, _mask in zip(_deltas[::-1], _dones[::-1]):
                _advantages.append(_delta + DISCOUNT_FACTOR * GAE_LAMBDA * _advantages[-1] * (1 - _mask))

            # Se invierte las ventajas almacenadas por eficiencia
            _advantages.reverse()
            _advantages = _advantages[:-1] # No tener en cuenta el 0 inicial
            _advantages = torch.tensor(_advantages).float().unsqueeze(1).to(self.critic.device)
            _returns = _advantages + _values
            # Se suma una pequeña constante para evitar errores en la división
            # Advantage scaling - Revisiting Design Choices in Proximal Policy Optimization
            # https://arxiv.org/abs/2009.10897
            _advantages = (_advantages - _advantages.mean()) / (_advantages.std() + _small_constant)

        return _advantages, _returns

    def learn(self):
        # Obtención de toda la memoria para calcular GAE
        _states, _next_states, _actions, _probabilities, _rewards, _dones = self.memory.get_all()

        # Conversión de los arrays obtenidos de la memoria de tensores de Pytorch
        _states = torch.tensor(_states, dtype=torch.float).to(self.critic.device)
        _next_states = torch.tensor(_next_states, dtype=torch.float).to(self.critic.device)
        _actions = torch.tensor(_actions, dtype=torch.float).to(self.critic.device)
        _probabilities = torch.tensor(_probabilities, dtype=torch.float).to(self.critic.device)
        _rewards = torch.tensor(_rewards, dtype=torch.float).unsqueeze(1).to(self.critic.device)

        # Get GAE
        _advantages, _returns = self.get_advantage_and_returns((_states, _next_states, _rewards, _dones))

        # Learn from batches per each epoch
        for _ in range(EPOCHS):
            self._learn_from_batches(
                batches=self.memory.get(),
                advantages=_advantages,
                returns=_returns,
                states=_states,
                probabilities=_probabilities,
                actions=_actions,
            )
            
        # Vaciado de la memoria
        self.memory.clear()

    def _learn_from_batches(self, batches, advantages, returns, states, actions, probabilities):
        for _batch in batches:
            _states = states[_batch]
            _actions = actions[_batch]
            _probabilities = probabilities[_batch]

            # En base a los estados, obtengo la distribución de las acciones del ator
            _distribution = self.actor(_states)
            # Obtención de las nuevas probabilidades
            _new_probabilities = _distribution.log_prob(_actions)
            # Calculo de la diferencia entre las nuevas probabilidades y las anteriores
            _probability_difference = torch.exp(_new_probabilities.sum(1, keepdim=True) - _probabilities.sum(1, keepdim=True))

            # Probabilidades ponderadas con los advantages
            _weighted_probabilities = advantages[_batch] * _probability_difference
            # Probabilidades ponderadas limitadas por la policy clip para evitar pasos
            # que difieran demasiado
            _weighted_clipped_probabilities = torch.clamp(
                _probability_difference, 1 - POLICY_CLIP, 1 + POLICY_CLIP
            ) * advantages[_batch]

            # Entropia de la distribución
            _entropy = _distribution.entropy().sum(1, keepdims=True)
            # La perdida del actor es el minimo entre las probabiliades y las clipeadas
            _actor_loss = -torch.min(_weighted_probabilities, _weighted_clipped_probabilities)
            # Y al valor anterior se le resta el coeficiente de entropia multiplicado por la entropia
            _actor_loss -= ENTROPY_COEFFICIENT * _entropy

            # Propagación hacia atras con limitación del gradiente
            self.actor.optimizer.zero_grad()
            _actor_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRADIENT_CLIPPING)
            self.actor.optimizer.step()

            # Una vez se ha realizado backpropagation con el actor, se procede con el crítico
            _critic_value = self.critic(_states)
            _critic_loss = (_critic_value - returns[_batch]).pow(2).mean()
            self.critic.optimizer.zero_grad()
            _critic_loss.backward()
            self.critic.optimizer.step()

    def add_to_memory(self, state, next_state, action, probabilities, reward, done):
        self.memory.add(state, next_state, action, probabilities, reward, done)

    def save_actor_critic(self):
        self.actor.save()
        self.critic.save()

    def load_actor_critic(self):
        self.actor.load()
        self.critic.load()
