from typing import Callable

import numpy

from dsfaker import InfiniteGenerator


class RandomNumber(InfiniteGenerator):
    def __init__(self, random_state_method: Callable, params: dict, dtype: numpy.dtype=None):
        self.method = random_state_method
        self.method_params = params
        self.dtype = dtype

    def get_single(self):
        return self.method(*self.method_params)

    def stream_single(self):
        while True:
            yield self.get_single()

    def get_batch(self, batch_size: int):
        return numpy.array(self.method(*self.method_params, size=(1, batch_size)), dtype=self.dtype)

    def stream_batch(self, batch_size: int):
        while True:
            yield numpy.array(self.get_batch(batch_size), dtype=self.dtype)


class BoundedRandomNumber(RandomNumber):
    """
    Bounded RandomNumber
    """
    def __init__(self, random_state_method: Callable, params: dict, lb=None, up=None):
        super().__init__(random_state_method, params)
        self.lb = lb
        self.up = up
    #TODO
