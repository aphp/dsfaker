from numpy.random.mtrand import RandomState


class Distribution:
    bounded = None
    continuous = None

    def get(self, size=None):
        raise NotImplementedError("get not implemented!")


class BoundedDistribution(Distribution):
    lb = None
    up = None
    bounded = True


class Beta(BoundedDistribution):
    """
    Beta law:
    * bounded
    * continuous

    .. math:: f(x; a,b) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 - x)^{\beta - 1},

    where the normalisation, B, is the beta function,

    .. math:: B(\alpha, \beta) = \int_0^1 t^{\alpha - 1} (1 - t)^{\beta - 1} dt.
    """
    continuous = True
    lb = 0
    up = 1

    def __init__(self, a, b, seed=None):
        self.a = a
        self.b = b
        self.rs = RandomState(seed=seed)

    def get(self, size=None):
        return self.rs.beta(self.a, self.b, size=size)


class Power(BoundedDistribution):
    """
    Power law:
    * bounded
    * continuous

    .. math:: P(x; a) = ax^{a-1}, 0 \le x \le 1, a>0.
    """
    continuous = True

    def __init__(self, a, seed=None):
        self.a = a
        self.rs = RandomState(seed=seed)

    def get(self, size=None):
        return self.rs.power(self.a, size=size)


