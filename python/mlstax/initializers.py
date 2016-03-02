"""
author : John Allard (github: jhallard)
date : 2/14/2016
LICENSE : MIT
"""
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import time

class Initializer(object) :
    """ 
    base class for the initializer hierarchy, each class defines a specific
    way to initialize the hidden weights for a given layer
    """
    def __init__(self) :
        self.rng = rng = numpy.random.RandomState(2702) # we can't use 2701, that's the product
                                                        # of two primes!

    def init_weights(self, size) :
        """
        override this function to return a matrix of dimensions `size` filled
        with the specific way you want to initialize it.
        """
        pass

def Uniform(Initializer) :
    """
    initializes a weight matrix from a uniform distribution, bounded above and below by
    wmax and wmin repectively.
    """
    def __init__(self, wmax, wmin) :
        super(Uniform, self).__init__(self)
        self.wmax = wmax
        self.wmin = wmin

    def init_weights(self, size) :
        return np.asarray(
            self.rng.uniform(
                low=self.wmin,
                high=self.wmax,
                size=size
            ),
            dtype=theano.config.floatX
        )



def Normal(Initializer) :
    """
    initializes a weight matrix from a uniform distribution, bounded above and below by
    wmax and wmin repectively.
    """
    def __init__(self, mean, stddev) :
        super(Uniform, self).__init__(self)
        self.mean = mean
        self.stddev = stdev

    def init_weights(self, size) :
        return np.asarray(
            self.rng.normal(
                loc=self.mean,
                scale=self.stddev,
                size=size
            ),
            dtype=theano.config.floatX
        )


def Zeros(Initializer) :
    """
    initializes a weight matrix to all zeros (pointless class I know)
    """
    def __init__(self) :
        super(Uniform, self).__init__(self)

    def init_weights(self, size) :
        return np.zeros(size, dtype=theano.config.floatX)


def Ones(Initializer) :
    """
    initializes a weight matrix to all ones (pointless class I know)
    """
    def __init__(self) :
        super(Uniform, self).__init__(self)

    def init_weights(self, size) :
        return np.ones(size, dtype=theano.config.floatX)
