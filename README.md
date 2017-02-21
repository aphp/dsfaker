# DataScienceFaker

[![Build Status][travis-image]][travis-url]
[![codecov][codecov-image]][codecov-url]
[![doc][doc-image]][doc-url]

DSFaker aims at providing data scientists with functions for generating and streaming arrays of data such as:

- Time-series
- Date/Datetime
- Autoincrement & random autoincrement
- Trigonometric (Sin, Sinh, Cos, Cosh, Tan, Tanh)
- Random distributions
- Pattern repetition
- Noise application

Basic operators between generators and/or python int/float types (+, -, /, //, \*, ...) are also implemented.

To generate such data, you can specify the distribution you want amongst the ones provided by numpy.random.
Or you can also implement your own distribution to generate data.

## Installation

```
pip install dsfaker
```

[travis-url]: https://travis-ci.org/Dubrzr/dsfaker
[travis-image]: https://travis-ci.org/Dubrzr/dsfaker.svg?branch=master

[codecov-url]: https://codecov.io/gh/Dubrzr/dsfaker
[codecov-image]: https://codecov.io/gh/Dubrzr/dsfaker/branch/master/graph/badge.svg

[doc-url]: https://dubrzr.github.io/dsfaker/
[doc-image]: https://img.shields.io/badge/docs-latest-brightgreen.svg
