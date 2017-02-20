from typing import Union, Iterable

import numpy
from numpy import ndarray
from numpy.random.mtrand import RandomState

from . import DistributionNonNegative, DistributionBounded, DistributionUnbounded


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
    ub = 1

    def __init__(self,
                 a: Union[float, ndarray, Iterable[float]],
                 b: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.a = a
        self.b = b
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.beta(a=self.a,
                            b=self.b,
                            size=size)


class Binomial(DistributionNonNegative):
    """
    The Binomial distribution is non-negative and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.binomial <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.binomial.html>`_.

    Distribution function:

    .. math:: P(N) = \\binom{n}{N}p^N(1-p)^{n-N}
    """
    continuous = False

    def __init__(self,
                 n: Union[int, ndarray, Iterable[int]],
                 p: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.n = n
        self.p = p
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.binomial(n=self.n,
                                p=self.p,
                                size=size)


class BinomialNegative(DistributionNonNegative):
    """
    The negative Binomial distribution is non-negative and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.negative_binomial <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.negative_binomial.html>`_.

    Distribution function:

    .. math:: P(N;n,p) = \\binom{N+n-1}{n-1}p^{n}(1-p)^{N}
    """
    continuous = False

    def __init__(self,
                 n: Union[int, ndarray, Iterable[int]],
                 p: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.n = n
        self.p = p
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.negative_binomial(n=self.n,
                                         p=self.p,
                                         size=size)


class CauchyStandard(DistributionUnbounded):
    """
    The standard Cauchy distribution is unbounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.standard_cauchy <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.standard_cauchy.html>`_.

    Distribution function:

    .. math:: P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\bigl[ 1+ (\\frac{x-x_0}{\\gamma})^2 \\bigr] }
    """
    continuous = True

    def __init__(self, seed=None):
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.standard_cauchy(size=size)


class Chisquare(DistributionNonNegative):
    """
    The Chisquare distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.chisquare <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.chisquare.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)} x^{k/2 - 1} e^{-x/2}
    """
    continuous = True

    def __init__(self,
                 k: Union[int, ndarray, Iterable[int]],
                 seed=None):
        self.k = k
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.chisquare(df=self.k,
                                 size=size)


class ChisquareNonCentral(DistributionNonNegative):
    """
    The non-central Chisquare distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.noncentral_chisquare <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.noncentral_chisquare.html>`_.

    Distribution function:

    .. math:: P(x;df,nonc) = \\sum^{\\infty}_{i=0} \\frac{e^{-nonc/2}(nonc/2)^{i}}{i!} P_{Y_{df+2i}}(x)
    """
    continuous = True

    def __init__(self,
                 k: Union[int, ndarray, Iterable[int]],
                 nonc: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.k = k
        self.nonc = nonc
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.noncentral_chisquare(df=self.k,
                                            nonc=self.nonc,
                                            size=size)


class Dirichlet(DistributionNonNegative):
    """
    The Dirichlet distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.dirichlet <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.dirichlet.html>`_.

    Distribution function:

    .. math:: X \\approx \\prod_{i=1}^{k}{x^{\\alpha_i-1}_i}
    """
    continuous = True

    def __init__(self,
                 alpha: list,
                 seed=None):
        self.alpha = alpha
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.dirichlet(alpha=self.alpha,
                                 size=size)


class Exponential(DistributionNonNegative):
    """
    The Exponential distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.exponential <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.exponential.html>`_.

    Distribution function:

    .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta})
    """
    continuous = True

    def __init__(self,
                 beta: Union[int, ndarray, Iterable[int]] = 1.0,
                 seed=None):
        self.beta = beta
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.exponential(scale=self.beta,
                                   size=size)


class F(DistributionNonNegative):
    """
    The Fisher distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.f <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.f.html>`_.
    """
    continuous = True

    def __init__(self,
                 dfnum: Union[int, ndarray, Iterable[int]],
                 dfden: Union[int, ndarray, Iterable[int]],
                 seed=None):
        self.dfnum = dfnum
        self.dfden = dfden
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.f(dfnum=self.dfnum,
                         dfden=self.dfden,
                         size=size)


class FNonCentral(DistributionNonNegative):
    """
    The non-central Fisher distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.noncentral_f <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.noncentral_f.html>`_.
    """
    continuous = True

    def __init__(self,
                 dfnum: Union[int, ndarray, Iterable[int]],
                 dfden: Union[int, ndarray, Iterable[int]],
                 nonc: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.dfnum = dfnum
        self.dfden = dfden
        self.nonc = nonc
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.noncentral_f(dfnum=self.dfnum,
                                    dfden=self.dfden,
                                    nonc=self.nonc,
                                    size=size)


class Gamma(DistributionNonNegative):
    """
    The Gamma distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.gamma <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.gamma.html>`_.

    Distribution function:

    .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)}
    """
    continuous = True

    def __init__(self,
                 k: Union[float, ndarray, Iterable[float]],
                 theta: Union[float, ndarray, Iterable[float]] = 1.0,
                 seed=None):
        self.k = k
        self.theta = theta
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.gamma(shape=self.k,
                             scale=self.theta,
                             size=size)


class Geometric(DistributionNonNegative):
    """
    The Geometric distribution is non-negative and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.geometric <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.geometric.html>`_.

    Distribution function:

    .. math:: f(k) = (1 - p)^{k - 1} p
    """
    continuous = False
    lb = 1

    def __init__(self,
                 p: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.p = p
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.geometric(p=self.p,
                                 size=size)


class Gumbel(DistributionUnbounded):
    """
    The Gumbel distribution is unbounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.gumbel <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.gumbel.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta} e^{ -e^{-(x - \\mu)/ \\beta}}
    """
    continuous = True

    def __init__(self,
                 mu: Union[float, ndarray, Iterable[float]] = 0.0,
                 beta: Union[float, ndarray, Iterable[float]] = 1.0,
                 seed=None):
        self.mu = mu
        self.beta = beta
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.gumbel(loc=self.mu,
                              scale=self.beta,
                              size=size)


class Hypergeometric(DistributionNonNegative):
    """
    The Hypergeometric distribution is non-negative and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.hypergeometric <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.hypergeometric.html>`_.

    Distribution function:

    .. math:: P(x) = \\frac{\\binom{m}{n}\\binom{N-m}{n-x}}{\\binom{N}{n}}
    """
    continuous = False

    def __init__(self,
                 n: int,
                 m: Union[int, ndarray, Iterable[int]],
                 N: Union[int, ndarray, Iterable[int]],
                 seed=None):
        self.n = n
        self.m = m
        self.N = N
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.hypergeometric(ngood=self.n,
                                      nbad=self.m,
                                      nsample=self.N,
                                      size=size)


class Laplace(DistributionUnbounded):
    """
    The Laplace distribution is unbounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.laplace <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.laplace.html>`_.

    Distribution function:

    .. math:: f(x; \\mu, \\lambda) = \\frac{1}{2\\lambda} \\exp\\left(-\\frac{|x - \\mu|}{\\lambda}\\right)
    """
    continuous = True

    def __init__(self,
                 mu: Union[float, ndarray, Iterable[float]],
                 beta: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.mu = mu
        self.beta = beta
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.laplace(loc=self.mu,
                               scale=self.beta,
                               size=size)


class Logistic(DistributionUnbounded):
    """
    The Logistic distribution is unbounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.logistic <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.logistic.html>`_.

    Distribution function:

    .. math:: P(x) = P(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2}
    """
    continuous = True

    def __init__(self,
                 mu: Union[float, ndarray, Iterable[float]],
                 beta: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.mu = mu
        self.beta = beta
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.logistic(loc=self.mu,
                                scale=self.beta,
                                size=size)


class Lognormal(DistributionNonNegative):
    """
    The Lognormal distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.lognormal <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.lognormal.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{1}{\\sigma x \\sqrt{2\\pi}} e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}
    """
    continuous = True

    def __init__(self,
                 mu: Union[float, ndarray, Iterable[float]],
                 sigma: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.mu = mu
        self.sigma = sigma
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.lognormal(mean=self.mu,
                                 sigma=self.sigma,
                                 size=size)


class Lomax(DistributionNonNegative):
    """
    The Pareto II or Lomax distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.pareto <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.pareto.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{am^a}{x^{a+1}}
    """
    continuous = True
    lb = 0

    def __init__(self,
                 a: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.a = a
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.pareto(a=self.a,
                              size=size)


class Multinomial(DistributionNonNegative):
    """
    The Multinomial distribution is non-negative and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.multinomial <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.multinomial.html>`_.
    """
    continuous = False

    def __init__(self,
                 n: int,
                 pvals: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.n = n
        self.pvals = pvals
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.multinomial(n=self.n,
                                   pvals=self.pvals,
                                   size=size)


class Normal(DistributionUnbounded):
    """
    The Normal/Gaussian distribution if unbounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.normal <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.normal.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }} e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} }
    """
    continuous = True

    def __init__(self,
                 mean: Union[float, ndarray, Iterable[float]] = 0.0,
                 std: Union[float, ndarray, Iterable[float]] = 1.0,
                 seed=None):
        self.mean = mean
        self.std = std
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.normal(loc=self.mean, scale=self.std, size=size)


class NormalMultivariate(DistributionUnbounded):
    """
    The multivariate Normal distribution is unbounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.multivariate_normal <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.multivariate_normal.html>`_.
    """
    continuous = False

    def __init__(self,
                 mu: Union[float, ndarray, Iterable[float]],
                 cov: Union[list, ndarray],
                 seed=None):
        self.mu = mu
        self.cov = cov
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.multivariate_normal(mean=self.mu,
                                           cov=self.cov,
                                           size=size)


class Poisson(DistributionNonNegative):
    """
    The Poisson distribution is non-negative and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.poisson <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.poisson.html>`_.

    Distribution function:

    .. math:: f(k; \\lambda)=\\frac{\\lambda^k e^{-\\lambda}}{k!}
    """
    continuous = False

    def __init__(self,
                 lam: Union[float, ndarray, Iterable[float]] = 1.0,
                 seed=None):
        self.lam = lam
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.poisson(lam=self.lam,
                               size=size)


class Power(DistributionBounded):
    """
    The Power distribution is bounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.power <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.power.html>`_.

    Distribution function:

    .. math:: P(x; a) = ax^{a-1}, 0 \\le x \\le 1, a>0.
    """
    continuous = True
    lb = 0
    ub = 1

    def __init__(self,
                 a: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.a = a
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.power(a=self.a,
                             size=size)


class Randint(DistributionBounded):
    """
    The Randint distribution is bounded and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.randint <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.randint.html>`_.
    """
    continuous = False

    def __init__(self,
                 lb: int,
                 ub: int,
                 seed=None):
        self.lb = lb
        self.ub = ub
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.randint(low=self.lb,
                               high=self.ub,
                               size=size)


class RandomSample(DistributionBounded):
    """
    The RandomSample distribution is bounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.random_sample <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.random_sample.html>`_.
    """
    continuous = True
    lb = 0
    ub = 1

    def __init__(self, seed=None):
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.random_sample(size=size)


class Rayleigh(DistributionNonNegative):
    """
    The Rayleigh distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.rayleigh <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.rayleigh.html>`_.

    Distribution function:

    .. math:: P(x;scale) = \\frac{x}{scale^2}e^{\\frac{-x^2}{2 \\cdotp scale^2}}
    """
    continuous = True

    def __init__(self,
                 sigma: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.sigma = sigma
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.rayleigh(scale=self.sigma,
                                size=size)


class Triangular(DistributionBounded):
    """
    The Triangular distribution is bounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.triangular <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.triangular.html>`_.

    Distribution function:

    .. math:: P(x;l, m, r) = \\begin{cases} \\frac{2(x-l)}{(r-l)(m-l)}& \\text{for $l \\leq x \\leq m$}, \\frac{2(r-x)}{(r-l)(r-m)}& \\text{for $m \\leq x \\leq r$}, 0& \\text{otherwise}. \\end{cases}
    """
    continuous = True

    def __init__(self,
                 left: Union[float, ndarray, Iterable[float]],
                 mode: Union[float, ndarray, Iterable[float]],
                 right: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.left = left
        self.mode = mode
        self.right = right
        self.lb = left
        self.ub = right
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.triangular(left=self.left,
                                  mode=self.mode,
                                  right=self.right,
                                  size=size)


class Uniform(DistributionBounded):
    """
    The Uniform distribution is bounded and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.uniform <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.uniform.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{1}{b - a}
    """
    continuous = False

    def __init__(self,
                 lb: Union[float, ndarray, Iterable[float]],
                 ub: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.lb = lb
        self.ub = ub
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.uniform(low=self.lb,
                               high=self.ub,
                               size=size)


class Vonmises(DistributionBounded):
    """
    The Vonmises distribution is bounded and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.vonmises <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.vonmises.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{e^{\\kappa cos(x-\\mu)}}{2\\pi I_0(\\kappa)}
    """
    continuous = True
    lb = -numpy.math.pi
    ub = +numpy.math.pi

    def __init__(self,
                 mu: Union[float, ndarray, Iterable[float]],
                 kappa: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.mu = mu
        self.kappa = kappa
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.vonmises(mu=self.mu,
                                kappa=self.kappa,
                                size=size)


class Wald(DistributionNonNegative):
    """
    The Wald distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.wald <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.wald.html>`_.

    Distribution function:

    continuous = True
    .. math:: P(x;mean,scale) = \\sqrt{\\frac{scale}{2\\pi x^3}}e^ \\frac{-scale(x-mean)^2}{2\\cdotp mean^2x}
    """

    def __init__(self,
                 mu: Union[float, ndarray, Iterable[float]],
                 lam: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.mu = mu
        self.lam = lam
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.wald(mean=self.mu,
                            scale=self.lam,
                            size=size)


class Weibull(DistributionNonNegative):
    """
    The Weibull distribution is non-negative and continuous.

    The implementation is from `numpy.random.mtrand.RandomState.weibull <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.weibull.html>`_.

    Distribution function:

    .. math:: X = (-ln(U))^{1/a}
    """
    continuous = True

    def __init__(self,
                 a: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.a = a
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.weibull(a=self.a,
                               size=size)


class Zipf(DistributionNonNegative):
    """
    The Zipf distribution is non-negative and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.zipf <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.zipf.html>`_.

    Distribution function:

    .. math:: p(x) = \\frac{x^{-a}}{\\zeta(a)}
    """
    continuous = False
    lb = 1

    def __init__(self,
                 a: Union[float, ndarray, Iterable[float]],
                 seed=None):
        self.a = a
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.zipf(a=self.a,
                            size=size)


class Choice(DistributionBounded):
    """
    The Choice distribution is bounded and discrete.

    The implementation is from `numpy.random.mtrand.RandomState.choice <https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.random.mtrand.RandomState.choice.html>`_.
    """
    continuous = False

    def __init__(self,
                 probabilities: numpy.array,
                 seed=None):
        self.probabilities = probabilities
        self.a = numpy.arange(len(probabilities))
        self.lb = 0
        self.ub = len(probabilities) - 1
        self.rs = RandomState(seed=seed)

    def _get(self, size=None):
        return self.rs.choice(a=self.a, p=self.probabilities, size=size)
