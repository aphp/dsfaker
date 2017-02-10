from numpy.random.mtrand import RandomState


class Distribution:
    bounded = None
    continuous = None

    def get(self, size=None):
        raise NotImplementedError("get not implemented!")


class DistributionUnbounded(Distribution):
    bounded = False


class DistributionNonNegative(Distribution):
    bounded = True
    lb = None


class DistributionBounded(Distribution):
    bounded = True
    lb = None
    up = None


class Beta(DistributionBounded):
    """
    The Beta distribution is bounded and continuous.
    The implementation is from `numpy.random.mtrand.RandomState.beta <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.beta.html>`_.

    Distribution function:

    .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1} (1 - x)^{\\beta - 1}

    where the normalisation, B, is the beta function,

    .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1} (1 - t)^{\\beta - 1} dt.
    """
    continuous = True
    lb = 0
    up = 1

    def __init__(self, a, b, seed=None):
        self.a = a
        self.b = b
        self.rs = RandomState(seed=seed)

    def get(self, size=None):
        return self.rs.beta(a=self.a, b=self.b, size=size)


class Power(DistributionBounded):
    """
    The Power distribution is bounded and continuous.
    The implementation is from `numpy.random.mtrand.RandomState.power <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.power.html>`_.

    Distribution function:

    .. math:: P(x; a) = ax^{a-1}, 0 \\le x \\le 1, a>0.
    """
    continuous = True

    def __init__(self, a: float, seed=None):
        self.a = a
        self.rs = RandomState(seed=seed)

    def get(self, size=None):
        return self.rs.power(a=self.a, size=size)


class Normal(DistributionUnbounded):
    """
    The Normal/Gaussian distribution
    The implementation is from `numpy.random.mtrand.RandomState.normal <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.normal.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }} e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} }
    """
    continuous = True

    def __init__(self, mean: float=0, std: float=1.0, seed=None):
        self.mean = mean
        self.std = std
        self.rs = RandomState(seed=seed)

    def get(self, size=None):
        return self.rs.normal(loc=self.mean, scale=self.std, size=size)


class Gamma(DistributionNonNegative):
    """
    The Gamma distribution
    The implementation is from `numpy.random.mtrand.RandomState.gamma <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.gamma.html>`_.

    Distribution function:

    .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)}
    """
    continuous = True
    lb = 0

    def __init__(self, k: float, theta: float, seed=None):
        self.k = k
        self.theta = theta
        self.rs = RandomState(seed=seed)

    def get(self, size=None):
        return self.rs.gamma(shape=self.k, scale=self.theta, size=size)
