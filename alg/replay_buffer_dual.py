import random
import numpy as np

class Replay_Buffer():

    def __init__(self, size=5e4):
        self.memory_1 = []
        self.memory_2 = []
        self.maxsize = int(size)
        self.idx_1 = 0
        self.idx_2 = 0

    def add(self, episode, is_bad=False):
        """Assumes that <episode> is a list of transitions.
        
        If is_bad=True, store episode in memory_1, else store in memory_2.
        """
        if is_bad:
            for idx in range(len(episode)):
                self.add_1(episode[idx])
        else:
            for idx in range(len(episode)):
                self.add_2(episode[idx])

    def add_1(self, transition):
        if self.idx_1 >= len(self.memory_1):
            self.memory_1.append(transition)
        else:
            self.memory_1[self.idx_1] = transition
        self.idx_1 = (self.idx_1 + 1) % self.maxsize

    def add_2(self, transition):
        if self.idx_2 >= len(self.memory_2):
            self.memory_2.append(transition)
        else:
            self.memory_2[self.idx_2] = transition
        self.idx_2 = (self.idx_2 + 1) % self.maxsize

    def sample_batch(self, size):
        """Samples a batch of transitions.
        
        If memory_1 and memory_2 both have enough samples, 
        then samples size/2 from each.
        Otherwise, returns all contents of the smaller memory,
        and takes the remainder from the larger one.
        """
        half = int(size/2.0)

        if half <= len(self.memory_1) and half > len(self.memory_2):
            # Enough bad transitions but not enough good ones
            remainder = size - len(self.memory_2)
            n_from_mem1 = min(len(self.memory_1), remainder)
            return np.array( random.sample(self.memory_1, n_from_mem1) + self.memory_2 )
        elif half > len(self.memory_1) and half <= len(self.memory_2):
            # Not enough bad transitions but enough good ones
            remainder = size - len(self.memory_1)
            n_from_mem2 = min(len(self.memory_2), remainder)
            return np.array( self.memory_1 + random.sample(self.memory_2, n_from_mem2) )
        elif len(self.memory_1) < half and len(self.memory_2) < half:
            # Both not enough
            return np.array( self.memory_1 + self.memory_2 )
        else:
            return np.array( random.sample(self.memory_1, half) + random.sample(self.memory_2, half) )
