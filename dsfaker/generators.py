# -*- coding: utf-8 -*-


class Generator():
    def get_single(self):
        """
        A function that returns a single element.
        Not implemented.
        """
        raise NotImplementedError("get_single not implemented")

    def stream_single(self):
        """
        A generator function that returns element by element.
        Not implemented.
        """
        raise NotImplementedError("stream_single not implemented")

    def get_batch(self, batch_size: int):
        """
        A function that returns a single batch of elements.
        Not implemented.
        """
        raise NotImplementedError("get_batch not implemented")

    def stream_batch(self, batch_size: int):
        """
        A generator function that returns batches of elements.
        Not implemented.
        """
        raise NotImplementedError("stream_batch not implemented")


class FiniteGenerator(Generator):
    def get_all(self, *args, **kwargs):
        """
        A function that returns all the elements.
        Not implemented.
        """
        raise NotImplementedError("get_all not implemented")


class InfiniteGenerator(Generator):
    pass
