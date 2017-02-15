from dsfaker import AddOperator
from dsfaker.distributions import Normal
from dsfaker.generators import RandomNumber


class WhiteGaussianNoise(AddOperator):
    def __init__(self, generator, distribution=Normal()):
        super().__init__(generator, RandomNumber(distribution=distribution))
