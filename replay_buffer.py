import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(s)
            actions.append(a)
            next_states.append(s_)
            rewards.append(r)
            dones.append(d)

        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards).reshape(-1, 1),
            np.array(dones).reshape(-1, 1),
        )
