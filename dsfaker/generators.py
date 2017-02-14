# -*- coding: utf-8 -*-
from typing import Callable

import numpy

from .operators import AddOperator, SubOperator, TrueDivOperator, FloorDivOperator, MulOperator, PowOperator, \
    ModOperator, AndOperator, XorOperator, OrOperator


class Generator():
    def get_single(self) -> float:
        """
        A function that returns a single element.
        Not implemented.
        """
        raise NotImplementedError("get_single not implemented")

    def stream_single(self) -> float:
        while True:
            yield self.get_single()

    def get_batch(self, batch_size: int) -> numpy.array:
        """
        A function that returns a single batch of elements.
        Not implemented.
        """
        raise NotImplementedError("get_batch not implemented")

    def stream_batch(self, batch_size: int) -> numpy.array:
        while True:
            yield self.get_batch(batch_size=batch_size)

    def __add__(self, other):
        return AddOperator(self, other)

    def __sub__(self, other):
        return SubOperator(self, other)

    def __truediv__(self, other):
        return TrueDivOperator(self, other)

    def __floordiv__(self, other):
        return FloorDivOperator(self, other)

    def __mul__(self, other):
        return MulOperator(self, other)

    def __pow__(self, other):
        return PowOperator(self, other)

    def __mod__(self, other):
        return ModOperator(self, other)

    def __and__(self, other):
        return AndOperator(self, other)

    def __or__(self, other):
        return OrOperator(self, other)

    def __xor__(self, other):
        return XorOperator(self, other)


class FiniteGenerator(Generator):
    finite = True

    def get_all(self, *args, **kwargs) -> numpy.array:
        """
        A function that returns all the elements.
        Not implemented.
        """
        raise NotImplementedError("get_all not implemented")


class InfiniteGenerator(Generator):
    finite = False
    pass


class UniqueValueGenerator(InfiniteGenerator):
    def __init__(self, value):
        self.value = value

    def get_single(self) -> float:
        return self.value

    def get_batch(self, batch_size: int) -> numpy.array:
        return numpy.ones(batch_size) * self.value


class ApplyFunctionGenerator(InfiniteGenerator):
    def __init__(self, function, generator: InfiniteGenerator):
        self.function = function
        self.generator = generator

    def get_single(self) -> float:
        return self.function(self.generator.get_single())

    def get_batch(self, batch_size: int) -> numpy.array:
        return self.function(self.generator.get_batch(batch_size=batch_size))
