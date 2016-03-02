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

import activations, initializers

class Layer(object) :
    """
    Base class for the Layer hierarchy. Each class represents a single
    layer in a deep neural network. The layers are then chained together under
    a model class along with activation functions.
    """
    def __init__(self, size, init=None, activation=None) :
        """
        @size - nodes in the layer
        @init - initializer class
        @activation - activation class
        """
        self.lnum = -1
        self.lname = ""
        self.weights = None 
        self.bias = None
        self.size = size
        self.init = init if init else initializers.Uniform(-1, 1)
        self.activation = activation if activation else activations.Sigmoid()

    def feed(self, data) :
        """ returns a theano equation defining the output of this layer given data """
        pass

    def init_weights(self, indim) :
        self.weights = self.init.init_weights((self.size, indim))
        self.bias = initializers.Zeros().init_weights((self.size, 1))
        self.weights = theano.shared(value=self.weights, name='weights', borrow=True)
        self.bias = theano.shared(value=self.bias, name='bias', borrow=True)


class Dense(Layer) :

    def __init__(self, size, init=None, activation=None, **kwargs) :
        super(Dense, self).__init__(size, **kwargs)
        self.lname = "Dense"

    def feed(self, data) :
        return self.activation.activate(T.dot(self.weights, data) + self.bias)

