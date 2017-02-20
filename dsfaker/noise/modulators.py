import numpy

from dsfaker.generators import Normal, TimeSeries, Generator, BoundingOperator


class ModulatorLinearInterpolation(Generator):
    def __init__(self,
                 time_series: TimeSeries,
                 start_frequency: float=1,
                 modulating_generator: Generator=BoundingOperator(generator=Normal(std=0.1),
                                                                  lb=1/3,
                                                                  ub=3),
                 modulating_step: int=10):
        """
        A Modulator implementing a linear interpolation for missing values

        :param time_series: A TimeSeries instance to be modulated
        :param start_frequency: The starting frequency (current_frequency = start_frequency)
        :param modulating_generator: The generator to use to modulate the frequency (current_frequency += modulating_generator.get_single())
        :param modulating_step: number of values between two calls to the modulating_generator
        """
        self.time_series = time_series
        self.start_frequency = start_frequency
        self.modulating_generator = modulating_generator
        self.modulating_step = modulating_step

        self.current_frequency = start_frequency
        self.current_time_delay = None
        self.future_values = None
        self.future_time = None
        self.index = None

    def get_single(self) -> float:
        # TODO
        raise NotImplementedError()

    def get_batch(self, batch_size: int) -> numpy.array:
        # TODO
        raise NotImplementedError()
