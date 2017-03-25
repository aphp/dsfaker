# -*- coding: utf-8 -*-
import copy
from functools import reduce
import operator
from typing import Iterable

import numpy

from dsfaker.listeners import Listener


class Generator():
    def _get_single(self):
        raise NotImplementedError("_get_single not implemented")

    def _get_batch(self, batch_size: int):
        raise NotImplementedError("_get_batch not implemented")

    def get_single(self):
        """
        A function that returns a single element.
        """
        val = self._get_single()
        if hasattr(self, '_listeners'):
            for listener in self._listeners:
                listener.put_single(val)
        return val


    def stream_single(self) -> Iterable:
        while True:
            yield self.get_single()

    def get_batch(self, batch_size: int) -> numpy.array:
        """
        A function that returns a batch of elements.
        """
        try:
            vals = self._get_batch(batch_size=batch_size)
        except NotImplementedError:
            res = []
            for _ in range(batch_size):
                res.append(self._get_single())
            vals = numpy.asarray(res)
        if hasattr(self, '_listeners'):
            for listener in self._listeners:
                listener.put_batch(vals)
        return vals

    def stream_batch(self, batch_size: int) -> Iterable:
        while True:
            yield self.get_batch(batch_size=batch_size)

    def add_listener(self, listener: Listener):
        if not hasattr(self, '_listeners'):
            self._listeners = []
        self._listeners.append(listener)

    def copy(self):
        tmp = copy.deepcopy(self)
        tmp._listeners = []
        return tmp

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

    def __neg__(self):
        return MulOperator(self, -1)

    __radd__ = __add__
    __rmul__ = __mul__
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def __rsub__(self, other):
        return SubOperator(other, self)

    def __rtruediv__(self, other):
        return TrueDivOperator(other, self)

    def __rfloordiv__(self, other):
        return FloorDivOperator(other, self)

    def __rpow__(self, other):
        return PowOperator(other, self)

    def __rmod__(self, other):
        return ModOperator(other, self)



class BoundedGenerator(Generator):
    bounded = True
    lb = None
    ub = None


class ReduceOperator(Generator):
    def __init__(self, *generators, reduce_lambda):
        self.generators = generators
        self.reduce_lambda = reduce_lambda

    def _get_single(self) -> float:
        def _get_single(x):
            if isinstance(x, Generator):
                return x.get_single()
            return x
        return reduce(lambda a, b: self.reduce_lambda(_get_single(a), _get_single(b)), self.generators)

    def _get_batch(self, batch_size: int) -> numpy.array:
        def _get_batch(x, batch_size):
            if isinstance(x, Generator):
                return x.get_batch(batch_size=batch_size)
            return x
        return reduce(lambda a, b: self.reduce_lambda(_get_batch(a, batch_size=batch_size), _get_batch(b, batch_size=batch_size)), self.generators)


class AddOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.add)

    def __str__(self):
        return '+'


class SubOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.sub)

    def __str__(self):
        return '-'

class TrueDivOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.truediv)

    def __str__(self):
        return '/'


class FloorDivOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.floordiv)

    def __str__(self):
        return '//'


class MulOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.mul)

    def __str__(self):
        return '*'


class PowOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.pow)

    def __str__(self):
        return '**'


class ModOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.mod)

    def __str__(self):
        return '%'


class AndOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.and_)

    def __str__(self):
        return '&'


class OrOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.or_)

    def __str__(self):
        return '|'


class XorOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=operator.xor)

    def __str__(self):
        return '^'


class Distribution(Generator):
    bounded = None
    continuous = None
    lb = None
    ub = None

    def _get(self, size=None):
        raise NotImplementedError("_get not implemented!")

    def _get_single(self) -> float:
        return self._get()

    def _get_batch(self, batch_size: int) -> numpy.array:
        return self._get(size=batch_size)


class DistributionUnbounded(Distribution):
    bounded = False


class DistributionNonNegative(Distribution):
    bounded = True
    lb = 0


class DistributionBounded(Distribution, BoundedGenerator):
    bounded = True
