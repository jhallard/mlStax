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

class Layer(object) :
    """
    Base class for the Layer hierarchy. Each class represents a single
    layer in a deep neural network. The layers are then chained together under
    a model class along with activation functions.
    """
    def __init__(self, size, init=None, activation="sigmoid") :
        self.weights = None 
        self.bias = None
        self.state = None 
        self.size = size
        self.init = init if init else Uniform(-1, 1)
        self.activation = activation if activation else Sigmoid

    def feed(self, data) :
        """ returns a theano equation defining the output of this layer given data """
        pass

    def init_weights(self) :
        if self.init == "uniform" :
            self.weights = np.asarray(
                    rng.uniform(


class Dense(Layer) :

    def __init__(self, size, init, activation, **kwargs) :
        super(Dense, self).__init__(size, init activation, **kwargs)
        self.weights = np.array(

    def feed(self, data) :
        return self.activation(T.dot(dat, self.weights) + self.bias)

