from . import Generator


class TimeSeries(Generator):
    def __init__(self, time_gen: Generator, data_gen: Generator):
        self.time_gen = time_gen
        self.data_gen = data_gen

    def get_single(self):
        return self.time_gen.get_single(), self.data_gen.get_single()

    def get_batch(self, batch_size: int):
        return self.time_gen.get_batch(batch_size=batch_size), self.data_gen.get_batch(batch_size=batch_size)
