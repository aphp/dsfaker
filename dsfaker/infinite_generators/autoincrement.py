import numpy

from dsfaker import InfiniteGenerator
from dsfaker.infinite_generators import RandomNumber


class Autoincrement(InfiniteGenerator):
    def __init__(self, start: int, step: int, dtype: numpy.dtype):
        self.start = start
        self.offset = 0
        self.dtype = dtype
        self.step = step

    def get_single(self):
        self.offset += 1
        return self.start + self.offset * self.step

    def get_batch(self, batch_size: int):
        self.offset += batch_size
        return numpy.arange(start=self.start + (self.offset - batch_size) * self.step,
                            stop=self.start + self.offset * self.step,
                            step=self.step,
                            dtype=self.dtype)


class AutoincrementRandom(InfiniteGenerator):
    def __init__(self, start: int, rn: RandomNumber):
        """

        :param start: The value to start with
        :param rn: A RandomNumber or its child classes instance
        :param dtype: The data type to return
        """
        self.start = start
        self.rn = rn
        self.current_val = start

    def get_single(self):
        tmp = self.rn.get_single()
        old_val = self.current_val
        self.current_val += tmp
        return old_val

    def get_batch(self, batch_size: int):
        random_incremental = self.current_val + numpy.cumsum(self.rn.get_batch(batch_size=batch_size))
        self.current_val = random_incremental[-1]
        return random_incremental
