import numpy

from . import Generator


class Autoincrement(Generator):
    def __init__(self, start: float=0.0, step: float=1.0, dtype: numpy.dtype=numpy.int64):
        self.start = start
        self.offset = 0
        self.dtype = dtype
        self.step = step

    def get_single(self):
        self.offset += 1
        return self.start + (self.offset - 1) * self.step

    def get_batch(self, batch_size: int):
        self.offset += batch_size
        return numpy.arange(start=self.start + (self.offset - batch_size) * self.step,
                            stop=self.start + self.offset * self.step,
                            step=self.step,
                            dtype=self.dtype)


class AutoincrementWithGenerator(Generator):
    def __init__(self, start: float, generator: Generator):
        """

        :param start: The value to start with
        :param generator: A Generator
        """
        self.start = start
        self.generator = generator
        self.current_val = start

    def get_single(self):
        tmp = self.generator.get_single()
        old_val = self.current_val
        self.current_val += tmp
        return old_val

    def get_batch(self, batch_size: int):
        random_incremental = self.current_val + numpy.cumsum(self.generator.get_batch(batch_size=batch_size))
        random_incremental = numpy.insert(random_incremental, 0, self.current_val)
        self.current_val = random_incremental[-1]
        return random_incremental[:-1]

