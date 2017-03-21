from random import Random
from rstr import Rstr
from . import Generator

class Regex(Generator):
    def __init__(self, regex, seed=None):
        self.gen = Rstr(Random(seed))
        self.regex = regex

    def get_single(self):
        return self.gen.xeger(self.regex)
