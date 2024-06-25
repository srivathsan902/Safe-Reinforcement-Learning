from collections import deque
import pickle
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.size = 0

    def add(self, state, action, reward, next_state, cost, done):
        self.buffer.append((state, action, reward, next_state, cost, done))
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, cost, dones = zip(*batch)
        return states, actions, rewards, next_states, cost, dones
    
    def save(self, path):
        np.save(path, np.array(self.buffer, dtype=object))

    def load(self, path):
        self.buffer = deque(np.load(path, allow_pickle=True).tolist(), maxlen=self.buffer.maxlen)
    
# Define the PER replay buffer
class PERBuffer:
    def __init__(self, max_size, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, cost, done):
        transition = (state, action, reward, next_state, cost, done)
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        if self.size < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == 0:
            return [], [], np.array([], dtype=np.float32)
        
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        if probs.sum() == 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + self.epsilon
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'buffer': self.buffer,
                'priorities': self.priorities,
                'pos': self.pos,
                'size': self.size
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = data['buffer']
            self.priorities = data['priorities']
            self.pos = data['pos']
            self.size = data['size']

# import numpy as np

# class SumTree:
#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.tree = np.zeros(2 * max_size - 1)  # Binary heap tree
#         self.data = np.zeros(max_size, dtype=object)  # Stored data
#         self.pointer = 0

#     def add(self, priority, data):
#         index = self.pointer + self.max_size - 1
#         self.data[self.pointer] = data
#         self.update(index, priority)
#         self.pointer += 1
#         if self.pointer >= self.max_size:
#             self.pointer = 0

#     def update(self, index, priority):
#         change = priority - self.tree[index]
#         self.tree[index] = priority
#         while index != 0:
#             index = (index - 1) // 2
#             self.tree[index] += change

#     def retrieve(self, value):
#         parent_idx = 0
#         while True:
#             left_child_idx = 2 * parent_idx + 1
#             if left_child_idx >= len(self.tree):
#                 leaf_idx = parent_idx
#                 break
#             else:
#                 if value <= self.tree[left_child_idx]:
#                     parent_idx = left_child_idx
#                 else:
#                     value -= self.tree[left_child_idx]
#                     parent_idx = left_child_idx + 1
#         data_index = leaf_idx - self.max_size + 1
#         return leaf_idx, self.tree[leaf_idx], self.data[data_index]

#     @property
#     def total_priority(self):
#         return self.tree[0]

# class PERBuffer:
#     def __init__(self, max_size, alpha=0.6, beta=0.4, epsilon=1e-6):
#         self.max_size = max_size
#         self.alpha = alpha
#         self.beta = beta
#         self.epsilon = epsilon
#         self.buffer = []
#         self.tree = SumTree(max_size)
#         self.size = 0

#     def add(self, state, action, reward, next_state, cost, done):
#         transition = (state, action, reward, next_state, cost, done)
#         max_priority = self.tree.tree.max() if self.size > 0 else 1.0
#         self.buffer.append(transition)
#         self.tree.add(max_priority, transition)
#         self.size = min(self.size + 1, self.max_size)

#     def sample(self, batch_size):
#         batch = []
#         indices = []
#         priorities = []
#         segment = self.tree.total_priority / batch_size

#         for i in range(batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             value = np.random.uniform(a, b)
#             index, priority, data = self.tree.retrieve(value)
#             batch.append(data)
#             indices.append(index)
#             priorities.append(priority)

#         total = self.tree.total_priority
#         weights = (total * np.array(priorities) / self.size) ** (-self.beta)
#         weights /= weights.max()

#         return batch, indices, weights

#     def update_priorities(self, indices, errors):
#         for idx, error in zip(indices, errors):
#             priority = abs(error) + self.epsilon
#             self.tree.update(idx, priority)

#     def save(self, path):
#         with open(path, 'wb') as f:
#             pickle.dump({
#                 'buffer': self.buffer,
#                 'tree': self.tree,
#                 'size': self.size
#             }, f)

#     def load(self, path):
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
#             self.buffer = data['buffer']
#             self.tree = data['tree']
#             self.size = data['size']
