import numpy

from dsfaker import InfiniteGenerator
from dsfaker.distributions import Distribution, DistributionBounded
from dsfaker.exceptions import NotCompatibleDistributionException


class RandomNumber(InfiniteGenerator):
    def __init__(self, distribution: Distribution, dtype: numpy.dtype=None):
        self.distribution = distribution
        self.dtype = dtype

    def get_single(self):
        return self.distribution.get()

    def get_batch(self, batch_size: int):
        return numpy.asarray(self.distribution.get(size=(1, batch_size)), dtype=self.dtype)


class RandomNumberBounded(RandomNumber):
    """
    Bounded RandomNumber
    """
    def __init__(self, distribution: DistributionBounded, lb: int, ub: int, dtype: numpy.dtype=None):
        super().__init__(distribution, dtype)
        if distribution.bounded is False:
            raise NotCompatibleDistributionException("BoundedRandomNumber need a BoundedDistribution.")
        if lb >= ub:
            raise ValueError("lb should be less than up")
        self.lb = lb
        self.ub = ub
        self.coef = (ub - lb) / (self.distribution.up - self.distribution.lb)

    def get_single(self):
        return self.distribution.get() * self.coef + self.lb - self.distribution.lb

    def get_batch(self, batch_size: int):
        return numpy.asarray(self.distribution.get(size=(1, batch_size)), dtype=self.dtype) * self.coef + self.lb - self.distribution.lb
