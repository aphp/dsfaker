from dsfaker.generators import AddOperator, Normal


class WhiteGaussianNoise(AddOperator):
    def __init__(self, generator, distribution=Normal()):
        super().__init__(generator, distribution)
