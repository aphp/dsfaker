import numpy

from dsfaker import InfiniteGenerator
from dsfaker.distributions import Distribution, BoundedDistribution
from dsfaker.exceptions import NotCompatibleDistributionException


class RandomNumber(InfiniteGenerator):
    def __init__(self, distribution: Distribution, dtype: numpy.dtype=None):
        self.distribution = distribution
        self.dtype = dtype

    def get_single(self):
        return self.distribution.get()

    def stream_single(self):
        while True:
            yield self.get_single()

    def get_batch(self, batch_size: int):
        return numpy.array(self.distribution.get(size=(1, batch_size)), dtype=self.dtype)

    def stream_batch(self, batch_size: int):
        while True:
            yield self.get_batch(batch_size)


class BoundedRandomNumber(RandomNumber):
    """
    Bounded RandomNumber
    """
    def __init__(self, distribution: BoundedDistribution, lb: int, up: int, dtype: numpy.dtype=None):
        super().__init__(distribution, dtype)
        if distribution.bounded is False:
            raise NotCompatibleDistributionException("BoundedRandomNumber need a BoundedDistribution.")
        if lb >= up:
            raise ValueError("lb should be less or equal to up")
        self.lb = lb
        self.up = up
        self.coef = (up - lb) / (self.distribution.up - self.distribution.lb)

    def get_single(self):
        return self.distribution.get() * self.coef + self.lb - self.distribution.lb

    def stream_single(self):
        while True:
            yield self.get_single()

    def get_batch(self, batch_size: int):
        return numpy.array(self.distribution.get(size=(1, batch_size)), dtype=self.dtype) * self.coef + self.lb - self.distribution.lb

    def stream_batch(self, batch_size: int):
        while True:
            yield self.get_batch(batch_size)
