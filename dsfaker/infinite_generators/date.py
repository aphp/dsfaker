import numpy

from dsfaker.distributions import BoundedDistribution
from . import RandomNumberBounded, InfiniteGenerator


class RandomDate(InfiniteGenerator):
    def __init__(self, distribution: BoundedDistribution, start: numpy.datetime64, end: numpy.datetime64, unit):
        """
        A timezone-aware class to generate dates between start and end (inclusive) following a certain distribution

        :param distribution:
        :param start: The starting date (inclusive)
        :param end: The ending date (inclusive)
        :param unit: The time unit to use for the distribution ('Y', 'M', 'W', 'D', 'h', 'm', 's', 'us', 'ms', 'ns', 'ps', 'fs', 'as')
        """
        self.rnb = RandomNumberBounded(distribution=distribution, lb=0, ub=(end-start) / numpy.timedelta64(1, unit), dtype=numpy.float64)
        self.start = start
        self.end = end
        self.unit = unit
        self.td_unit = 'timedelta64[{}]'.format(unit)

    def get_single(self):
        return self.start + numpy.timedelta64(int(round(self.rnb.get_single())), self.unit)

    def stream_single(self):
        while True:
            yield self.get_single()

    def get_batch(self, batch_size: int):
        return self.start + self.rnb.get_batch(batch_size=batch_size).astype(self.td_unit)

    def stream_batch(self, batch_size: int):
        while True:
            yield self.get_batch(batch_size)
