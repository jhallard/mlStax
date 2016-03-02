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

class Activation(object) :
    """ 
    Class to wrap around a given activation function. Most of the ones
    I define will simply wrap a theano nnet function, but I'm making a class
    hierarchy so that people can submit custom functions and so we can stack
    them more easily
    """

    def activate(self, data) :
        """ override this to use your activation function on the given input """
        pass

class Sigmoid(Activation) :
    def activate(self, data) :
        return T.nnet.sigmoid(data)

class Tanh(Activation) :
    def activate(self, data) :
        return T.tanh(data)

class ReLU(Activation) :
    def activate(self, data) :
        return T.nnet.relu(data)

class Softmax(Activation) :
    def activate(self, data) :
        return T.nnet.softmax(data)

class Nothing(Activation) :
    def activate(self, data) :
        return data
