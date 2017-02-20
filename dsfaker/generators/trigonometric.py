import numpy

from . import ApplyFunctionOperator, BoundedGenerator, Generator, Autoincrement


class Trigo(ApplyFunctionOperator):
    def __init__(self, function, generator: Generator):
        super().__init__(function, generator)


class Sin(ApplyFunctionOperator, BoundedGenerator):
    def __init__(self, generator: Generator=Autoincrement()):
        super().__init__(numpy.sin, generator)
        self.lb = -1.0
        self.ub = 1.0


class Sinh(Trigo):
    def __init__(self, generator: Generator=Autoincrement()):
        super().__init__(numpy.sinh, generator)


class Cos(Trigo, BoundedGenerator):
    def __init__(self, generator: Generator=Autoincrement()):
        super().__init__(numpy.cos, generator)
        self.lb = -1.0
        self.ub = 1.0


class Cosh(Trigo):
    def __init__(self, generator: Generator=Autoincrement()):
        super().__init__(numpy.cosh, generator)


class Tan(Trigo):
    def __init__(self, generator: Generator=Autoincrement()):
        super().__init__(numpy.tan, generator)


class Tanh(Trigo, BoundedGenerator):
    def __init__(self, generator: Generator=Autoincrement()):
        super().__init__(numpy.tanh, generator)
        self.lb = -1.0
        self.ub = 1.0
