import numpy as np
import pytest
from dsfaker.generators import Autoincrement, RepeatPattern


class TestRepeatPattern:
    def test_type(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
        assert rp.get_single() == 0
        assert isinstance(rp.get_batch(9), np.ndarray)

    def test_shape(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
        assert rp.get_batch(5010).shape == (5010,)

    def test_len(self):
        rp = RepeatPattern([0,1,2,3,4,5,6,7,8,9])
        for i in range(10):
            nb = np.random.randint(2, 1000000)
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
    def test_len(self):
        pass

