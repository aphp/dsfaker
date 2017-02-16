import numpy

from dsfaker.exceptions import NotCompatibleDistributionException
from . import Distribution, InfiniteGenerator, Generator


class ConstantValueGenerator(InfiniteGenerator):
    def __init__(self, value, dtype: numpy.dtype):
        self.value = value
        self.dtype = dtype

    def get_single(self) -> float:
        return self.value

    def get_batch(self, batch_size: int) -> numpy.array:
        return numpy.ones(batch_size, dtype=self.dtype) * self.value


class BoundedGenerator(Generator):
    def __init__(self, generator: Generator, lb: float, ub: float):
        self.generator = generator
        self.lb = lb
        self.ub = ub

    def get_single(self):
        return numpy.clip(self.generator.get_single(), self.lb, self.ub)

    def get_batch(self, batch_size: int):
        return numpy.clip(self.generator.get_batch(batch_size=batch_size), self.lb, self.ub)


class ScaleOperator(Generator):
    def __init__(self, distribution: Distribution, lb: int, ub: int, dtype: numpy.dtype=None):
        if distribution.bounded is False:
            raise NotCompatibleDistributionException("BoundedRandomNumber need a BoundedDistribution.")
        if lb >= ub:
            raise ValueError("lb should be less than up")
        self.lb = lb
        self.ub = ub
        self.distribution = distribution
        self.dtype = dtype
        self.coef = (ub - lb) / (self.distribution.up - self.distribution.lb)

    def get_single(self):
        return self.distribution._get() * self.coef + self.lb - self.distribution.lb

    def get_batch(self, batch_size: int):
        return numpy.asarray(self.distribution._get(size=batch_size), dtype=self.dtype) * self.coef + self.lb - self.distribution.lb


class ApplyFunctionOperator(Generator):
    def __init__(self, function, generator: Generator):
        self.function = function
        self.generator = generator

    def get_single(self) -> float:
        return self.function(self.generator.get_single())

    def get_batch(self, batch_size: int) -> numpy.array:
        return self.function(self.generator.get_batch(batch_size=batch_size))


class AbsoluteOperator(ApplyFunctionOperator):
    def __init__(self, generator):
        super().__init__(numpy.absolute, generator)
