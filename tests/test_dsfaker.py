import datetime
from decimal import Decimal
import re

import numpy as np
import pytest

from dsfaker.generators import Generator, ScalingOperator, RandomDatetime, Distribution, \
    NotCompatibleGeneratorException, Beta, Binomial, BinomialNegative, CauchyStandard, Chisquare, ChisquareNonCentral, \
    Dirichlet, Exponential, F, FNonCentral, Gamma, Geometric, Gumbel, Hypergeometric, Laplace, Logistic, Lognormal, \
    Multinomial, NormalMultivariate, Normal, Lomax, Poisson, Power, Randint, RandomSample, Rayleigh, Triangular, \
    Uniform, Vonmises, Wald, Weibull, Zipf, DistributionUnbounded, DistributionBounded, DistributionNonNegative, Sinh, \
    Cosh, Tanh, Tan, BoundedGenerator, Choice, CastOperator, TimeDelayedGenerator, History, MeanHistory
from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
from dsfaker.generators.series import RepeatPattern
from dsfaker.generators.str import RegexGenerator
from dsfaker.generators.trigonometric import Sin, Cos
from dsfaker.generators.timeseries import TimeSeries
from dsfaker.generators.utils import ConstantValueGenerator, BoundingOperator, ApplyFunctionOperator, AbsoluteOperator


class TestGenerator:
    def test_raises(self):
        g = Generator()
        with pytest.raises(NotImplementedError):
            g.get_single()
        with pytest.raises(NotImplementedError):
            g.get_batch(200)

    def test_stream_single(self):
        g = Autoincrement()
        c = 0
        for i in range(1000):
            for val in g.stream_single():
                assert val == c
                c += 1
                if c > 100000:
                    break

    def test_stream_batch(self):
        g = Autoincrement()
        c = 0
        for i in range(10):
            nb = np.random.randint(2, 1000)
            for values in g.stream_batch(nb):
                for val in values:
                    assert val == c
                    c += 1
                if c > 100000:
                    break

    def test_copy(self):
        g1 = Autoincrement()
        g2 = g1.copy()

        g2.get_single()

        assert g1.get_single() == g2.get_single() - 1



    def _get_two_unique_gen(self):
        va = np.random.randint(-1000, +1000, dtype=np.int32)
        vb = np.random.randint(0, 10000, dtype=np.int32)
        a = ConstantValueGenerator(va, dtype=np.int64)
        b = ConstantValueGenerator(vb, dtype=np.int64)
        return va, vb, a, b

    def test_add_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a + b
        assert c.get_single() == va + vb

    def test_add_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen()
        assert (2 + a).get_single() == 2 + va
        assert (b + 6).get_single() == vb + 6
        assert sum((7 + b).get_batch(10)) == 10 * (7 + vb)

    def test_sub_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a - b
        assert c.get_single() == va - vb

    def test_sub_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen()
        assert (2 - a).get_single() == 2 - va
        assert (b - 6).get_single() == vb - 6

    def test_truediv_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a / b
        assert c.get_single() == va / vb

    def test_truediv_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen()
        assert (2 / a).get_single() == 2 / va
        assert (b / 6).get_single() == vb / 6

    def test_floordiv_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a // b
        assert c.get_single() == va // vb

    def test_floordiv_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen()
        assert (2 // a).get_single() == 2 // va
        assert (b // 6).get_single() == vb // 6

    def test_mul_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a * b
        assert c.get_single() == va * vb

    def test_mul_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen()
        assert (2 * a).get_single() == 2 * va
        assert (b * 6).get_single() == vb * 6

    def test_pow_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a ** b
        assert c.get_single() == va ** vb

    def test_pow_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen()
        assert (2 ** b).get_single() == 2 ** vb
        assert (a ** 6).get_single() == va ** 6

    def test_mod_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a % b
        assert c.get_single() == va % vb

    def test_mod_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen()
        assert (2 % a).get_single() == 2 % va
        assert (b % 6).get_single() == vb % 6

    def _get_two_unique_gen_binary(self):
        va = np.random.randint(0, 2, dtype=np.int32)
        vb = np.random.randint(0, 2, dtype=np.int32)
        a = ConstantValueGenerator(va, dtype=np.int64)
        b = ConstantValueGenerator(vb, dtype=np.int64)
        return va, vb, a, b

    def test_and_op(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        c = a & b
        assert c.get_single() == va & vb

    def test_and_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        assert (1 & b).get_single() == 1 & vb
        assert (a & 1).get_single() == va & 1
        assert (0 & b).get_single() == 0 & vb
        assert (a & 0).get_single() == va & 0

    def test_or_op(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        c = a | b
        assert c.get_single() == va | vb

    def test_or_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        assert (1 | b).get_single() == 1 | vb
        assert (a | 1).get_single() == va | 1
        assert (0 | b).get_single() == 0 | vb
        assert (a | 0).get_single() == va | 0

    def test_xor_op(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        c = a ^ b
        assert c.get_single() == va ^ vb

    def test_xor_op_scalar(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        assert (1 ^ b).get_single() == 1 ^ vb
        assert (a ^ 1).get_single() == va ^ 1
        assert (0 ^ b).get_single() == 0 ^ vb
        assert (a ^ 0).get_single() == va ^ 0

    def test_neg_op(self):
        v = np.random.randint(-1000, +1000, dtype=np.int32)
        c = - ConstantValueGenerator(v, dtype=np.int64)
        assert c.get_single() == - v


class TestBoundingOperator:
    def test_values_single(self):
        n = Sin() * ConstantValueGenerator(50, dtype=np.uint16)
        lb = 0
        ub = 0
        while lb >= ub:
            lb = np.random.randint(-20, 20)
            ub = np.random.randint(-20, 20)
        bg = BoundingOperator(n, lb=lb, ub=ub)
        for i in range(10000):
            assert lb <= bg.get_single() <= ub

    def test_values_batch(self):
        n = Sin() * ConstantValueGenerator(50, dtype=np.uint16)
        lb = 0
        ub = 0
        while lb >= ub:
            lb = np.random.randint(-20, 20)
            ub = np.random.randint(-20, 20)
        bg = BoundingOperator(n, lb=lb, ub=ub)
        for i in range(10):
            nb = np.random.randint(2, 10000)
            values = bg.get_batch(nb)
            for val in values:
                assert lb <= val <= ub


class TestScalingOperator:
    def test_raises(self):
        with pytest.raises(ValueError):
            ScalingOperator(generator=None, lb=10, ub=0, dtype=np.int8)
        with pytest.raises(NotCompatibleGeneratorException):
            ScalingOperator(generator=Generator(), lb=0, ub=10, dtype=np.int8)

    def test_values_single(self):
        triangular_fun = BoundingOperator(
            ApplyFunctionOperator(function=lambda x: abs((x % 4) - 2) - 1, generator=Autoincrement()), lb=-1, ub=1)
        n = ScalingOperator(generator=triangular_fun, lb=-10, ub=10, dtype=np.float32)
        for _ in range(10000):
            assert n.get_single() == 10
            assert n.get_single() == 0
            assert n.get_single() == -10
            assert n.get_single() == 0

    def test_values_batch(self):
        triangular_fun = BoundingOperator(
            ApplyFunctionOperator(function=lambda x: abs((x % 4) - 2) - 1, generator=Autoincrement()), lb=-1, ub=1)
        n = ScalingOperator(generator=triangular_fun, lb=-10, ub=10, dtype=np.float32)
        tmp = [10, 0, -10, 0]
        count = 0
        for i in range(10):
            nb = np.random.randint(2, 1000)
            values = n.get_batch(nb)
            for j, val in enumerate(values):
                assert val == tmp[(count + j) % 4]
            count += nb


class TestApplyFunctionOperator:
    def _get_afo(self, fun):
        generator = Autoincrement()
        afo = ApplyFunctionOperator(function=fun, generator=generator)
        return afo

    def test_type(self):
        afo = self._get_afo(lambda x: 2 * x)
        assert afo.get_single() == 0
        assert isinstance(afo.get_batch(9), np.ndarray)

    def test_shape(self):
        afo = self._get_afo(lambda x: 2 * x)
        nb = np.random.randint(2, 10000)
        assert afo.get_batch(nb).shape == (nb,)

    def test_values_single(self):
        afo = self._get_afo(lambda x: 4 * x + 2)
        for i in range(10000):
            assert afo.get_single() == 4 * i + 2

    def test_values_batch(self):
        afo = self._get_afo(lambda x: 4 * x + 2)
        count = 0
        for i in range(10):
            nb = np.random.randint(2, 10000)
            values = afo.get_batch(nb)
            for j, val in enumerate(values):
                assert val == (count + j) * 4 + 2
            count += nb


class TestAbsoluteOperator:
    def test_values_single(self):
        n = Cos() * ConstantValueGenerator(50, dtype=np.uint16)
        ao = AbsoluteOperator(n)
        for i in range(10000):
            assert ao.get_single() >= 0

    def test_values_batch(self):
        n = Cos() * ConstantValueGenerator(50, dtype=np.uint16)
        ao = AbsoluteOperator(n)
        for i in range(10):
            nb = np.random.randint(2, 10000)
            values = ao.get_batch(nb)
            for val in values:
                assert val >= 0


class TestRepeatPattern:
    def test_type(self):
        rp = RepeatPattern([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert rp.get_single() == 0
        assert isinstance(rp.get_batch(9), np.ndarray)

    def test_shape(self):
        rp = RepeatPattern([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        nb = np.random.randint(2, 10000)
        assert rp.get_batch(nb).shape == (nb,)

    def test_len(self):
        rp = RepeatPattern([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        for i in range(10):
            nb = np.random.randint(2, 100000)
            print(nb)
            assert len(rp.get_batch(nb)) == nb

    def test_values_single(self):
        rp = RepeatPattern([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        for i in range(10000):
            assert rp.get_single() == i % 10

    def test_values_batch(self):
        rp = RepeatPattern([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        idx = 0
        for _ in range(10):
            nb = np.random.randint(2, 10000)
            print(nb)
            for val in rp.get_batch(nb):
                assert val == idx % 10
                idx += 1


class TestAutoincrement:
    def test_type(self):
        ai = Autoincrement(start=-42, step=+42, dtype=np.int32)
        assert ai.get_single() == -42
        assert isinstance(ai.get_batch(42), np.ndarray)

    def test_shape(self):
        ai = Autoincrement(start=-42, step=+42, dtype=np.int32)
        nb = np.random.randint(2, 10000)
        assert ai.get_batch(nb).shape == (nb,)

    def test_values_single(self):
        ai = Autoincrement(start=-42, step=+42, dtype=np.int32)
        for i in range(np.random.randint(10, 10000)):
            val = ai.get_single()
            assert val == -42 + i * 42

    def test_values_batch(self):
        ai = Autoincrement(start=-4.2, step=+4.2, dtype=np.float64)
        count = 0
        for i in range(10):
            nb = np.random.randint(2, 10000)
            values = ai.get_batch(nb)
            for j, val in enumerate(values):
                d = (Decimal(-4.2) + ((Decimal(count + j)) * Decimal(4.2)))
                assert round(Decimal(val), 1) == round(d, 1)
            count += nb


class TestAutoincrementRandom:
    def _get_gen(self):
        start = np.random.randint(-50, +50)
        step = np.random.randint(-10, 10)
        gen = ConstantValueGenerator(step, dtype=np.int32)
        ai = AutoincrementWithGenerator(start=start, generator=gen)
        return start, step, ai

    def test_type(self):
        start, step, ai = self._get_gen()
        assert ai.get_single() == start
        assert isinstance(ai.get_batch(42), np.ndarray)

    def test_shape(self):
        start, step, ai = self._get_gen()
        nb = np.random.randint(2, 100000)
        assert ai.get_batch(nb).shape == (nb,)

    def test_values_single(self):
        start, step, ai = self._get_gen()
        for i in range(np.random.randint(10, 10000)):
            val = ai.get_single()
            assert val == start + i * step

    def test_values_batch(self):
        start, step, ai = self._get_gen()
        count = 0
        for i in range(10):
            nb = np.random.randint(2, 10000)
            values = ai.get_batch(nb)
            for j, val in enumerate(values):
                assert val == start + (count + j) * step
            count += nb


class TestTimeSeries:
    def _get_ts(self):
        time = Autoincrement()
        data = ConstantValueGenerator(value=42, dtype=np.uint16)
        return TimeSeries(time_gen=time, data_gen=data)

    def test_type(self):
        ts = self._get_ts()
        assert ts.get_single() == (0, 42)
        assert isinstance(ts.get_batch(9)[0], np.ndarray)
        assert isinstance(ts.get_batch(9)[1], np.ndarray)

    def test_values_single(self):
        ts = self._get_ts()
        for i in range(1000):
            assert ts.get_single() == (i, 42)

    def test_values_batch(self):
        ts = self._get_ts()
        for i in range(10):
            tt, vv = ts.get_batch(100)
            for j, t in enumerate(tt):
                assert t == 100 * i + j
            for j, v in enumerate(vv):
                assert v == 42


class TestDate:
    def test_values_single(self):
        triangular_fun = BoundingOperator(
            ApplyFunctionOperator(function=lambda x: abs((x % 4) - 2) - 1, generator=Autoincrement()), lb=-1, ub=1)
        rd = RandomDatetime(generator=triangular_fun, start=np.datetime64("1950"), end=np.datetime64("2042"), unit="Y")
        for i in range(10000):
            assert rd.get_single() == np.datetime64('2042')
            assert rd.get_single() == np.datetime64('1996')
            assert rd.get_single() == np.datetime64('1950')
            assert rd.get_single() == np.datetime64('1996')

    def test_values_batch(self):
        triangular_fun = BoundingOperator(
            ApplyFunctionOperator(function=lambda x: abs((x % 4) - 2) - 1, generator=Autoincrement()), lb=-1, ub=1)
        rd = RandomDatetime(generator=triangular_fun, start=np.datetime64("1950"), end=np.datetime64("2042"), unit="Y")
        tmp = [np.datetime64('2042'), np.datetime64('1996'), np.datetime64('1950'), np.datetime64('1996')]
        count = 0
        for i in range(10):
            nb = np.random.randint(2, 1000)
            values = rd.get_batch(nb)
            for j, val in enumerate(values):
                assert val == tmp[(count + j) % 4]
            count += nb


class TestDistribution:
    def test_raise(self):
        with pytest.raises(NotImplementedError):
            d = Distribution()
            d.get_single()
        with pytest.raises(NotImplementedError):
            d = Distribution()
            d.get_batch(10)


class TestDistributions:
    def _get_all_distributions(self):
        distributions = [
            Beta(a=2, b=2),
            Binomial(n=20, p=0.5),
            BinomialNegative(n=20, p=0.5),
            CauchyStandard(),
            Chisquare(k=6),
            ChisquareNonCentral(k=4, nonc=1),
            Dirichlet(alpha=[0.3, 0.3, 0.3]),
            Exponential(),
            F(dfnum=5, dfden=2),
            FNonCentral(dfnum=10, dfden=20, nonc=5),
            Gamma(k=1.0),
            Geometric(p=0.5),
            Gumbel(),
            Hypergeometric(n=50, m=50, N=80),
            Laplace(mu=-5, beta=4),
            Logistic(mu=6, beta=2),
            Lognormal(mu=0, sigma=1.5),
            Lomax(a=2),
            Multinomial(n=10, pvals=[1 / 10 for _ in range(10)]),
            Normal(),
            NormalMultivariate(mu=[0, 0], cov=[[1, 0], [0, 100]]),
            Poisson(lam=4),
            Power(a=2),
            Randint(lb=-10, ub=20),
            RandomSample(),
            Rayleigh(sigma=1),
            Triangular(left=10, mode=22, right=42),
            Uniform(lb=-42, ub=84),
            Vonmises(mu=-1, kappa=1),
            Wald(mu=2, lam=0.2),
            Weibull(a=0.5),
            Zipf(a=2),
            Choice(probabilities=[.05, .15, .05, .20, .25, .10, .20])
        ]
        return distributions

    def test_attributes(self):
        for d in self._get_all_distributions():
            if isinstance(d, DistributionUnbounded):
                assert d.lb is None
                assert d.ub is None
            elif isinstance(d, DistributionNonNegative):
                assert d.lb is not None
                assert d.ub is None
            elif isinstance(d, DistributionBounded):
                assert d.lb is not None
                assert d.ub is not None

    def test_bounds_single(self):
        for d in self._get_all_distributions():
            if isinstance(d, DistributionBounded):
                for _ in range(10000):
                    v = d.get_single()
                    assert d.lb <= v <= d.ub
            elif isinstance(d, DistributionNonNegative):
                for _ in range(10000):
                    v = d.get_single()
                    if isinstance(v, np.ndarray):
                        for vv in v:
                            assert d.lb <= vv
                    else:
                        assert d.lb <= v
            elif isinstance(d, DistributionUnbounded):
                for _ in range(10000):
                    v = d.get_single()

    def test_bounds_batch(self):
        for d in self._get_all_distributions():
            if isinstance(d, DistributionBounded):
                for v in d.get_batch(10000):
                    assert d.lb <= v <= d.ub
            elif isinstance(d, DistributionNonNegative):
                for v in d.get_batch(10000):
                    if isinstance(v, np.ndarray):
                        for vv in v:
                            assert d.lb <= vv
                    else:
                        assert d.lb <= v
            elif isinstance(d, DistributionUnbounded):
                d.get_batch(10000)


class TestTrigo:
    def _get_all(self):
        functions = [
            Sin(),
            Sinh(),
            Cos(),
            Cosh(),
            Tan(),
            Tanh()
        ]
        return functions

    def test_bounds_single(self):
        for f in self._get_all():
            if isinstance(f, BoundedGenerator):
                for _ in range(10000):
                    assert f.lb <= f.get_single() <= f.ub
            else:
                for _ in range(10000):
                    f.get_single()

    def test_bounds_batch(self):
        for f in self._get_all():
            if isinstance(f, BoundedGenerator):
                for v in f.get_batch(10000):
                    assert f.lb <= v <= f.ub
            else:
                f.get_batch(10000)


class TestCastGenerator:
    def test_types(self):
        n = Normal()
        cg = CastOperator(generator=n, dtype=np.int16)
        cg.get_single()
        assert cg.get_batch(1000).dtype == np.int16


class TestTimeDelayedGenerator:
    def test_values_single(self):
        time_delay_sec = 0.005
        tdg = TimeDelayedGenerator(generator=ConstantValueGenerator(21, dtype=np.uint16), time_delay_sec=time_delay_sec)

        start_time = datetime.datetime.now()
        for _ in range(100):
            tdg.get_single()
        end_time = datetime.datetime.now()
        elapsed_timedelta = (end_time - start_time)
        assert datetime.timedelta(seconds=.47) <= elapsed_timedelta <= datetime.timedelta(seconds=.53)

        tdg = TimeDelayedGenerator(generator=ConstantValueGenerator(21, dtype=np.uint16),
                                   time_delay_generator=ConstantValueGenerator(time_delay_sec, dtype=np.float16))

        start_time = datetime.datetime.now()
        for _ in range(100):
            tdg.get_single()
        end_time = datetime.datetime.now()
        elapsed_timedelta = (end_time - start_time)
        assert datetime.timedelta(seconds=.47) <= elapsed_timedelta <= datetime.timedelta(seconds=.53)

    def test_values_batch(self):
        time_delay_sec = 0.0005
        tdg = TimeDelayedGenerator(generator=ConstantValueGenerator(21, dtype=np.uint16), time_delay_sec=time_delay_sec)

        start_time = datetime.datetime.now()
        for _ in range(10):
            tdg.get_batch(100)
        end_time = datetime.datetime.now()
        elapsed_timedelta = (end_time - start_time)
        assert datetime.timedelta(seconds=.47) <= elapsed_timedelta <= datetime.timedelta(seconds=.53)

        tdg = TimeDelayedGenerator(generator=ConstantValueGenerator(21, dtype=np.uint16),
                                   time_delay_generator=ConstantValueGenerator(time_delay_sec, dtype=np.float16))

        start_time = datetime.datetime.now()
        for _ in range(10):
            tdg.get_batch(100)
        end_time = datetime.datetime.now()
        elapsed_timedelta = (end_time - start_time)
        assert datetime.timedelta(seconds=.47) <= elapsed_timedelta <= datetime.timedelta(seconds=.53)


class TestHistory:
    def test_values_single(self):
        gen = History(Autoincrement(), 42)
        for i in range(10):
            assert i == gen.get_single()
        for i in range(10):
            assert gen.get_prev(-10+i) == i
        for i in range(10):
            assert i + 10 == gen.get_single()
        for i in range(10):
            assert gen.get_prev(-10+i) == i + 10

    def test_values_batch(self):
        gen = History(Autoincrement(), 42)

        for i, v in enumerate(gen.get_batch(10)):
            assert i == v
        for i in range(10):
            assert gen.get_prev(-10+i) == i


class TestMeanHistory:
    def test_values_single(self):
        gen = MeanHistory(Autoincrement(start=4), 4, initial_values=[0,1,2,3])

        for i in range(42):
            assert gen.get_single() == (i * 4.0 + 6.0) / 4.0

    def test_values_batch(self):
        gen = MeanHistory(Autoincrement(start=4), 4, initial_values=[0,1,2,3])

        for i in range(42):
            for j, v in enumerate(gen.get_batch(10)):
                assert v == ((i * 10 + j) * 4.0 + 6.0) / 4.0


class TestRegex:
    def test_values_single(self):
        patterns = [r'(0|\+33|0033)[1-9][0-9]{8}']

        for pattern in patterns:
            gen = RegexGenerator(pattern)

            for i in range(42):
                assert re.fullmatch(pattern, gen.get_single()) is not None

    def test_values_batch(self):
        patterns = [r'(0|\+33|0033)[1-9][0-9]{8}']

        for pattern in patterns:
            gen = RegexGenerator(pattern)

            for i in range(42):
                for e in gen.get_batch(10):
                    assert re.fullmatch(pattern, gen.get_single()) is not None
