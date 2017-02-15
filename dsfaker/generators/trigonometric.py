import numpy

from . import ApplyFunctionOperator
from . import InfiniteGenerator, Autoincrement


class Trigo(ApplyFunctionOperator):
    def __init__(self, function, generator: InfiniteGenerator):
        super().__init__(function, generator)


class Sin(Trigo):
    def __init__(self, generator: InfiniteGenerator=Autoincrement(0, 1, dtype=numpy.int64)):
        super().__init__(numpy.sin, generator)


class Sinh(Trigo):
    def __init__(self, generator: InfiniteGenerator=Autoincrement(0, 1, dtype=numpy.int64)):
        super().__init__(numpy.sinh, generator)


class Cos(Trigo):
    def __init__(self, generator: InfiniteGenerator=Autoincrement(0, 1, dtype=numpy.int64)):
        super().__init__(numpy.cos, generator)


class Cosh(Trigo):
    def __init__(self, generator: InfiniteGenerator=Autoincrement(0, 1, dtype=numpy.int64)):
        super().__init__(numpy.cosh, generator)


class Tan(Trigo):
    def __init__(self, generator: InfiniteGenerator=Autoincrement(0, 1, dtype=numpy.int64)):
        super().__init__(numpy.tan, generator)


class Tanh(Trigo):
    def __init__(self, generator: InfiniteGenerator=Autoincrement(0, 1, dtype=numpy.int64)):
        super().__init__(numpy.tanh, generator)