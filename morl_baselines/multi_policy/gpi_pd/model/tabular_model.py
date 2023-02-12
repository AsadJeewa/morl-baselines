import random

import numpy as np

from morl_baselines.common.prioritized_buffer import SumTree


class TabularModel:
    """
    Tabular dynamics model S_{t+1}, R_t ~ m(.,.|s,a)
    """

    def __init__(self, deterministic: bool = False, prioritize=False, max_size=int(1e5)) -> None:
        self.deterministic = deterministic
        self.model = dict()
        self.state_actions_pairs = list()
        self.prioritize = prioritize
        if self.prioritize:
            self.priorities = SumTree(max_size=max_size)
            self.sa_to_ind = dict()

    def update(self, state, action, reward, next_state, terminal, priority=None):
        sa = (tuple(state), int(action))
        srt = (tuple(next_state), tuple(reward) if isinstance(reward, np.ndarray) else reward, terminal)

        if sa not in self.model:
            self.state_actions_pairs.append(sa)
            if priority is not None:
                self.priorities.set(len(self.state_actions_pairs) - 1, priority)
                self.sa_to_ind[sa] = len(self.state_actions_pairs) - 1

            if self.deterministic:
                self.model[sa] = srt
            else:
                self.model[sa] = {srt: 1}
        else:
            if priority is not None:
                self.priorities.set(self.sa_to_ind[sa], priority)

            if not self.deterministic:
                self.model[sa][srt] = self.model[sa].get(srt, 0) + 1

    def predict(self, state, action):
        sa = (tuple(state), int(action))
        if sa not in self.model:
            return None, None, None

        if self.deterministic:
            next_state, reward, terminal = self.model[sa]
        else:
            next = list(self.model[sa].keys())
            probs = np.array(list(self.model[sa].values()), dtype=np.float32)
            next_state, reward, terminal = random.choices(next, weights=probs / probs.sum(), k=1)[0]
        if isinstance(reward, tuple):
            reward = np.array(reward)
        return next_state, reward, terminal

    def transitions(self, state, action):
        sa = (tuple(state), int(action))
        if sa not in self.model:
            return [((None, None, None), None)]

        if self.deterministic:
            next_state, reward, terminal = self.model[sa]
            return [((next_state, reward, terminal), 1.0)]
        else:
            next = list(self.model[sa].keys())
            probs = np.array(list(self.model[sa].values()), dtype=np.float32)
            probs /= probs.sum()
            return list(zip(next, probs))

    def probs(self, state, action):
        sa = (tuple(state), int(action))
        if self.deterministic or sa not in self.model:
            return [1.0]
        probs = np.array(list(self.model[sa].values()), dtype=np.float32)
        probs /= probs.sum()
        return probs

    def random_transition(self):
        if self.prioritize:
            ind = self.priorities.sample(1)[0]
        else:
            ind = random.randint(0, len(self.state_actions_pairs) - 1)
        sa = self.state_actions_pairs[ind]
        if self.deterministic:
            srt = self.model[sa]
            if self.prioritize:
                return sa[0], sa[1], np.array(srt[1]), srt[0], srt[2], ind
            else:
                return sa[0], sa[1], np.array(srt[1]), srt[0], srt[2]  # S A R S T
        else:
            next = list(self.model[sa].keys())
            probs = np.array(list(self.model[sa].values()))
            next_state, reward, terminal = random.choices(next, weights=probs / probs.sum(), k=1)[0]
            if isinstance(reward, tuple):
                reward = np.array(reward)
            if self.prioritize:
                return sa[0], sa[1], reward, next_state, terminal, ind
            else:
                return sa[0], sa[1], reward, next_state, terminal

    def update_priority(self, ind, priority):
        self.priorities.set(ind, priority)
        if ind > 0:
            self.priorities.set(ind - 1, max(priority, self.priorities.get_priority(ind - 1)))
