import numpy

from dsfaker import InfiniteGenerator


class Autoincrement(InfiniteGenerator):
    def __init__(self, start: int, dtype: numpy.dtype, step: int=1):
        self.start = start
        self.offset = 0
        self.dtype = dtype
        self.step = step

    def _produce_single(self):
        self.offset += 1
        return self.start + self.offset * self.step

    def reset(self, start: int=None, step: int=None):
        self.offset = 0
        if start is not None:
            self.start = start
        if step is not None:
            self.step = step

    def get_single(self):
        return self._produce_single()

    def stream_single(self):
        while True:
            yield self._produce_single()

    def get_batch(self, batch_size: int):
        self.offset += batch_size
        return numpy.arange(start=self.start + (self.offset - batch_size) * self.step,
                            stop=self.start + self.offset * self.step,
                            step=self.step,
                            dtype=self.dtype)

    def stream_batch(self, batch_size: int):
        while True:
            yield self.get_batch(batch_size)
