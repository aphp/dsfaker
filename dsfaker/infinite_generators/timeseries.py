import numpy

from dsfaker import InfiniteGenerator


class TimeSeries(InfiniteGenerator):
    pass


class TimeSeriesFromPattern(TimeSeries):
    def __init__(self, pattern: numpy.array):
        self.pattern = pattern
