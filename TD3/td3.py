import torch as torch
import torch.nn.functional as functional

from config import (BATCH_SIZE, DISCOUNT_FACTOR, NOISE_CLIPPING, POLICY_NOISE,
                    POLICY_UPDATE_FREQUENCY, TAU)
from memory import Memory
from networks.actor import Actor
from networks.critic import Critic


class TD3:
    def __init__(self, states_dimension, actions_dimension, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(states_dimension, actions_dimension, max_action)
        self.actor_target = Actor(states_dimension, actions_dimension, max_action)

        self.critic = Critic(states_dimension, actions_dimension)
        self.critic_target = Critic(states_dimension, actions_dimension)

        self.memory = Memory()
        self.max_action = max_action

    def select_action(self, state):
        with torch.no_grad():
            _state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        
        return self.actor(_state).cpu().data.numpy().flatten()

    def learn(self, iterations):
        for _it in range(iterations):
            # Extraemos un conjunto de transiciones de la memoria
            _states, _next_states, _actions, _rewards, _dones = self.memory.get(BATCH_SIZE)

            # Lo cargamos en tensores de PyTorch
            _states_tensor = torch.Tensor(_states).to(self.device)
            _next_states_tensor = torch.Tensor(_next_states).to(self.device)
            _actions_tensor = torch.Tensor(_actions).to(self.device)
            _rewards_tensor = torch.Tensor(_rewards).to(self.device)
            _dones_tensor = torch.Tensor(_dones).to(self.device)
            
            # Dado el estado siguiente el actor target ejecuta la siguiente acción
            _next_action = self.actor_target(_next_states_tensor)

            # Se añade ruido gausiano limitado por el hiper parámetro noise clipping y 
            # por la maxima acción permitida por el entorno
            _noise = torch.Tensor(_actions).data.normal_(0, POLICY_NOISE).to(self.device) 
            _noise_limited = _noise.clamp(-NOISE_CLIPPING, NOISE_CLIPPING)
            _next_action = (_next_action + _noise_limited).clamp(-self.max_action, self.max_action)

            # Dado un estado y accion siguientes, los target devuelven los valores Q asociados
            _output_q_value_1, _output_q_value_2 = self.critic_target(_next_states_tensor, _next_action)

            # El mínimo de ambos valores Q es el valor aproximado del siguiente estado
            _target_q_value = torch.min(_output_q_value_1, _output_q_value_2)

            # Se calcula el valor final incluyendo la recompensa y el valor gamma.
            _target_q_value = _rewards_tensor + ((1-_dones_tensor) * DISCOUNT_FACTOR * _target_q_value).detach()

            # Los dos críticos toman estados y acciones actuales y devuelven los valores Q
            _current_q_value_1, _current_q_value_2 = self.critic(_states_tensor, _actions_tensor)

            # Se calcula la pérdida del modelo como la suma de los dos MSE
            _critic_loss = functional.mse_loss(_current_q_value_1, _target_q_value) + functional.mse_loss(_current_q_value_2, _target_q_value)

            # Propagación hacia atrás del crítico
            self.critic.optimizer.zero_grad()
            _critic_loss.backward()
            self.critic.optimizer.step()

            # Cada POLICY_UPDATE_FREQUENCY iteraciones se actualiza el actor ejecutando gradient ascent en el output del primer crítico.
            if _it % POLICY_UPDATE_FREQUENCY == 0:
                _actor_loss = -self.critic.forward_Q_value_1(_states_tensor, self.actor(_states_tensor)).mean()
                self.actor.optimizer.zero_grad()
                _actor_loss.backward()
                self.actor.optimizer.step()

                # Actualización de los pesos del actor target considerando TAU
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

                # Actualización de los pesos del critico target considerando TAU
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

    def add_to_memory(self, state, next_state, action, reward, done):
        self.memory.add((state, next_state, action, reward, done))

    def save_actor_critic(self):
        self.actor.save()
        self.critic.save()

    def load_actor_critic(self):
        self.actor.load()
        self.critic.load()
