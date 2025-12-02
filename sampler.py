from abc import ABC, abstractmethod
import random
from typing import Optional


class Sampler(ABC):
    @abstractmethod
    def nextReflection(self, pReflect: float) -> bool:
        pass

    @abstractmethod
    def copy(self):
        pass


class RandomSampler(Sampler):
    def __init__(self, seed: int, pool_size: int = 100):
        self.pool_size = int(pool_size)
        self.index = 0
        random.seed(seed)
        self.pool = [random.random() for _ in range(self.pool_size)]
        self.decisions: list[bool] = []

    def nextReflection(self, pReflect: float) -> bool:
        r = self.pool[self.index % self.pool_size]
        self.index += 1
        decision = r < float(pReflect)
        self.decisions.append(decision)
        return decision
    
    # copy of random sampler will ensure that the same decisions are made
    def copy(self):
        return RefractiveMaskSampler(self.decisions)


class DeterministicSampler(Sampler):
    def nextReflection(self, pReflect: float) -> bool:
        return pReflect >= 1.0
    
    def copy(self):
        return DeterministicSampler()
    
    def reverse(self):
        return DeterministicSampler()


class RefractiveMaskSampler(Sampler):
    def __init__(self, mask: list[bool]):
        self.mask = mask
        self.index = 0

    def nextReflection(self, pReflect: float) -> bool:
        if self.index >= len(self.mask):
            return pReflect >= 1.0  # default behavior when mask is exhausted (prefer refract if possible)
        result = self.mask[self.index]
        self.index += 1
        return result
    
    def copy(self):
        return RefractiveMaskSampler(self.mask)
    
    def reverse(self):
        # reverse the mask, but extract the last element and insert it as the last element after reversing
        # The idea is that we originally started at the camera, until we hit P. However, the hit with P is also saved as a decision in this list.
        # When reversing, we start at P until we hit the camera, so the first surface decision we do, will be the one BEFORE P (so mask[-2] and not mask[-1]).
        if len(self.mask) <= 0:
            return RefractiveMaskSampler(self.mask)
        last_element = self.mask[-1]
        newMask = self.mask[:-1][::-1] + [last_element]
        return RefractiveMaskSampler(newMask)