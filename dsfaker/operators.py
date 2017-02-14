from functools import reduce
import operator

import numpy

from . import ApplyFunctionGenerator
from . import Generator


class ReduceOperator(Generator):
    def __init__(self, *generators, reduce_lambda):
        self.generators = generators
        self.reduce_lambda = reduce_lambda

    def get_single(self) -> float:
        return reduce(lambda a, b: self.reduce_lambda(a.get_single(), b.get_single()), self.generators)

    def get_batch(self, batch_size: int) -> numpy.array:
        return reduce(lambda a, b: self.reduce_lambda(a.get_batch(batch_size=batch_size), b.get_batch(batch_size=batch_size)), self.generators)


class AddOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.add(a,b))


class SubOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.sub(a,b))


class TrueDivOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.truediv(a,b))


class FloorDivOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.floordiv(a,b))


class MulOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.mul(a,b))


class PowOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.pow(a,b))


class ModOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.mod(a,b))


class AndOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.and_(a,b))


class OrOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.or_(a,b))


class XorOperator(ReduceOperator):
    def __init__(self, *generators):
        super().__init__(*generators, reduce_lambda=lambda a, b: operator.xor(a,b))


class AbsoluteOperator(ApplyFunctionGenerator):
    def __init__(self, generator):
        super().__init__(numpy.absolute, generator)
