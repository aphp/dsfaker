import math
import numpy

from dsfaker import InfiniteGenerator


class Serie(InfiniteGenerator):
    pass


class RepeatPattern(Serie):
    def __init__(self, pattern: numpy.array):
        self.pattern = pattern
        self.l = len(pattern)
        self.index = 0

    def _get(self, size):
        if size == 1:
            if self.index == self.l:
                self.index = 1
                return self.pattern[0]
            else:
                self.index += 1
                return self.pattern[self.index - 1]
        else:
            if self.index == self.l:
                self.index = 0

            remaining = self.l - self.index

            if size > remaining:
                nb_tiles = math.ceil((size - remaining) / self.l)
            else:
                nb_tiles = 0

            if nb_tiles > 0:
                tiles = numpy.tile(self.pattern, nb_tiles)
                split_index = size - (nb_tiles * self.l) - remaining
                old_idx = self.index
                self.index = self.l + split_index
                return numpy.concatenate((self.pattern[old_idx:], tiles[:split_index] if split_index != 0 else tiles))
            else:
                self.index += size
                return self.pattern[(self.index-size):self.index-1]

    def get_single(self):
        return self._get(1)

    def stream_single(self):
        while True:
            yield self.get_single()

    def get_batch(self, batch_size: int):
        return self._get(batch_size)

    def stream_batch(self, batch_size: int):
        while True:
            yield self.get_batch(batch_size)
