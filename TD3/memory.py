import numpy as np

from config import BUFFER_SIZE


class Memory:
    def __init__(self):
        self.memory_items = []
        self.ptr = 0

    def add(self, transition):
        if len(self.memory_items)== BUFFER_SIZE:
            self.memory_items[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % BUFFER_SIZE
        else:
            self.memory_items.append(transition)

    def get(self, batch_size):
        _indexes = np.random.randint(0, len(self.memory_items), size = batch_size)

        _states = []
        _next_states = []
        _actions = []
        _rewards = []
        _dones = []
        
        for _index in _indexes:
            _state, _next_state, _action, _reward, _done = self.memory_items[_index]

            _states.append(np.array(_state, copy = False))
            _next_states.append(np.array(_next_state, copy = False))
            _actions.append(np.array(_action, copy = False))
            _rewards.append(np.array(_reward, copy = False))
            _dones.append(np.array(_done, copy = False))
            
        return np.array(_states), \
            np.array(_next_states), \
            np.array(_actions), \
            np.array(_rewards).reshape(-1, 1), \
            np.array(_dones).reshape(-1, 1)

