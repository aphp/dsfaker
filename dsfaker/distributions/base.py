class Distribution:
    bounded = None
    continuous = None

    def get(self, size=None):
        raise NotImplementedError("get not implemented!")


class DistributionUnbounded(Distribution):
    bounded = False


class DistributionNonNegative(Distribution):
    bounded = True
    lb = 0


class DistributionBounded(Distribution):
    bounded = True
    lb = None
    up = None
