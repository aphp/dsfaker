from dsfaker import InfiniteGenerator


class TimeSeries(InfiniteGenerator):
    def __init__(self, data_gen: InfiniteGenerator, time_gen: InfiniteGenerator):
        self.data_gen = data_gen
        self.time_gen = time_gen

    def get_single(self):
        return self.data_gen.get_single(), self.time_gen.get_single()

    def get_batch(self, batch_size: int):
        return self.data_gen.get_batch(batch_size=batch_size), self.time_gen.get_batch(batch_size=batch_size)
