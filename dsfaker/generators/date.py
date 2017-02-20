import numpy

from . import Generator, ScalingOperator, BoundedGenerator


class RandomDatetime(Generator):
    def __init__(self, generator: BoundedGenerator, start: numpy.datetime64, end: numpy.datetime64, unit):
        """
        A timezone-aware class to generate datetimes between start and end (inclusive) following a certain distribution

        :param start: The starting date (inclusive)
        :param end: The ending date (inclusive)
        :param unit: The time unit to use for the distribution ('Y', 'M', 'W', 'D', 'h', 'm', 's', 'us', 'ms', 'ns', 'ps', 'fs', 'as')
        """
        self.rnb = ScalingOperator(generator=generator, lb=0, ub=(end - start) / numpy.timedelta64(1, unit), dtype=numpy.float64)
        self.start = start
        self.end = end
        self.unit = unit
        self.td_unit = 'timedelta64[{}]'.format(unit)

    def get_single(self):
        return self.start + numpy.timedelta64(int(round(self.rnb.get_single())), self.unit)

    def get_batch(self, batch_size: int):
        return self.start + self.rnb.get_batch(batch_size=batch_size).astype(self.td_unit)
