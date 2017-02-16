import numpy as np
from decimal import Decimal

import pytest

from dsfaker.generators import Generator, FiniteGenerator, ScalingOperator
from dsfaker.generators.distributions import Normal
from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
from dsfaker.generators.series import  RepeatPattern
from dsfaker.generators.trigonometric import Sin, Cos
from dsfaker.generators.timeseries import  TimeSeries
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

    def test_sub_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a - b
        assert c.get_single() == va - vb

    def test_truediv_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a / b
        assert c.get_single() == va / vb

    def test_floordiv_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a // b
        assert c.get_single() == va // vb

    def test_mul_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a * b
        assert c.get_single() == va * vb

    def test_pow_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a ** b
        assert c.get_single() == va ** vb

    def test_mod_op(self):
        va, vb, a, b = self._get_two_unique_gen()
        c = a % b
        assert c.get_single() == va % vb

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

    def test_or_op(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        c = a | b
        assert c.get_single() == va | vb

    def test_xor_op(self):
        va, vb, a, b = self._get_two_unique_gen_binary()
        c = a ^ b
        assert c.get_single() == va ^ vb


class TestFiniteGenerator:
    def test_raises(self):
        fg = FiniteGenerator()
        with pytest.raises(NotImplementedError):
            fg.get_all()


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
    def test_values_single(self):
        triangular_fun = BoundingOperator(ApplyFunctionOperator(function=lambda x: abs((x % 4)-2)-1, generator=Autoincrement()), lb=-1, ub=1)
        n = ScalingOperator(generator=triangular_fun, lb=-10, ub=10, dtype=np.float32)
        for _ in range(10000):
            assert n.get_single() == 10
            assert n.get_single() == 0
            assert n.get_single() == -10
            assert n.get_single() == 0


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


class TestRandomNumber:
    def test_type(self):
        distrib = Normal()
        assert distrib.get_single()
