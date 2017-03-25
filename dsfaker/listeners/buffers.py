import numpy

from dsfaker.listeners.base import Listener


class CircularBuffer(Listener):
    def __init__(self, size, initial_values=None):
        self.size = size
        if initial_values is None:
            self.buff = numpy.zeros(size, dtype=numpy.float64)
        else:
            assert size == len(initial_values)
            self.buff = numpy.asarray(initial_values)
        self.idx = 0

    def _put(self, value):
        if self.idx == self.size:
            self.idx = 0
        self.buff[self.idx] = value
        self.idx += 1

    def put_single(self, value):
        self._put(value=value)

    def put_batch(self, values):
        for value in values:
            self._put(value=value)

    def get_prev(self, i):
        return self.buff[(self.size + self.idx + i) % self.size]

    def get_all(self):
        return numpy.roll(self.buff, -self.idx)

    def get_mean(self):
        return numpy.sum(self.buff) / self.size
