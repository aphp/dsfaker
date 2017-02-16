import numpy

from dsfaker.exceptions import NotCompatibleGeneratorException
from . import BoundedGenerator, InfiniteGenerator, Generator, DistributionBounded


class ConstantValueGenerator(InfiniteGenerator):
    def __init__(self, value, dtype: numpy.dtype):
        self.value = value
        self.dtype = dtype

    def get_single(self) -> float:
        return self.value

    def get_batch(self, batch_size: int) -> numpy.array:
        return numpy.ones(batch_size, dtype=self.dtype) * self.value


class BoundingOperator(BoundedGenerator):
    def __init__(self, generator: Generator, lb: float, ub: float):
        self.generator = generator
        self.lb = lb
        self.ub = ub

    def get_single(self):
        return numpy.clip(self.generator.get_single(), self.lb, self.ub)

    def get_batch(self, batch_size: int):
        return numpy.clip(self.generator.get_batch(batch_size=batch_size), self.lb, self.ub)


class ScalingOperator(BoundedGenerator):
    def __init__(self, generator: BoundedGenerator, lb: float, ub: float, dtype: numpy.dtype=None):
        if lb >= ub:
            raise ValueError("lb should be less than ub")

        if not isinstance(generator, BoundedGenerator):
            raise NotCompatibleGeneratorException("ScalingOperator needs a BoundedGenerator.")

        self.generator = generator
        self.lb = lb
        self.ub = ub
        self.mid = ub - (ub - lb) / 2
        self.gen_mid =  self.generator.ub - (self.generator.ub - self.generator.lb) / 2
        self.coef = (ub - lb) / (self.generator.ub - self.generator.lb)
        self.dtype = dtype

    def get_single(self):
        return self.generator.get_single() * self.coef - (self.gen_mid * self.coef) + self.mid

    def get_batch(self, batch_size: int):
        return numpy.asarray(self.generator.get_batch(batch_size=batch_size), dtype=self.dtype) * self.coef - (self.gen_mid * self.coef) + self.mid


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
