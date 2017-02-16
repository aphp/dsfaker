import numpy as np
from decimal import Decimal

from dsfaker.generators import Autoincrement, AutoincrementWithGenerator, RepeatPattern, RandomNumber, UniqueValueGenerator


class TestRepeatPattern:
    def test_type(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
        assert rp.get_single() == 0
        assert isinstance(rp.get_batch(9), np.ndarray)

    def test_shape(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
        nb = np.random.randint(2, 10000)
        assert rp.get_batch(nb).shape == (nb,)

    def test_len(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
        for i in range(10):
            nb = np.random.randint(2, 100000)
            print(nb)
            assert len(rp.get_batch(nb)) == nb

    def test_values_single(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
        for i in range(10000):
            assert rp.get_single() == i % 10

    def test_values_batch(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
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
            for i, val in enumerate(values):
                d = (Decimal(-4.2) + ((Decimal(count + i)) * Decimal(4.2)))
                assert round(Decimal(val), 1) == round(d, 1)
            count += nb


class TestAutoincrementRandom:
    def _get_gen(self):
        start = np.random.randint(-50, +50)
        step = np.random.randint(-10, 10)
        gen = UniqueValueGenerator(step, dtype=np.int32)
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
            for i, val in enumerate(values):
                assert val == start + (count + i) * step
            count += nb

