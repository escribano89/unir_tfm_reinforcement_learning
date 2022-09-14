import numpy as np


# Memoria de experiencias actuales para PPO
class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.probabilities = []
        self.dones = [] # Flag que indica que ha terminado el episodio
        self.batch_size = batch_size

    # Devolver toda la memoria almacenada para calcular 
    # la función de ventaja para la trajectoria entera
    def get_all(self):
        return np.array(self.states),\
            np.array(self.next_states),\
            np.array(self.actions),\
            np.array(self.probabilities),\
            np.array(self.rewards),\
            np.array(self.dones)

    # Obtengo batches de tamaño batch_size utilizando la reorganización de los
    # indices aleatoria, de forma que para 8 elementos almacenados y un batch size de dos
    # se podría obtener: [array([7, 4]), array([0, 3]), array([2, 5]), array([1, 6])]
    # Idea obtenida de Phil Tabor
    # https://github.com/philtabor/Advanced-Actor-Critic-Methods/blob/main/PPO/single/continuous/memory.py
    def _get_batches(self, indexes, number_of_batches):
        _batches = []

        for _i in range(number_of_batches):
            _batches.append(indexes[_i * self.batch_size:(_i + 1) * self.batch_size])

        return _batches

    # Obtención de batches
    def get(self):
        _number_of_states = len(self.states)
        _indexes = np.arange(_number_of_states, dtype=np.int64)

        # Reorganizar los indices de forma aleatoria
        np.random.shuffle(_indexes)

        return self._get_batches(_indexes, int(_number_of_states // self.batch_size))

    # Añadir una experiencia a la memoria
    def add(self, state, next_state, action, probabilities, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.probabilities.append(probabilities)
        self.rewards.append(reward)
        self.dones.append(done)
    
    # Vaciar la memoria
    def clear(self):
        self.states = []
        self.next_states = []
        self.probabilities = []
        self.actions = []
        self.rewards = []
        self.dones = []
