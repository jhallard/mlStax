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

def mse(out, labels) :
    error = out - labels
    return T.dot(error, error.T)

class Optimizer(object) :
    """
    defines a algorithm/method for optimizing the parameters of
    a neural network. The most common is stochastic gradient descent,
    but there exist more advanced methods that can also be used. 
    """

    def __init__(self) :
        pass


    def updates(self, layers, output, costfn) :
        """
        returns a list of theano shared variable updates to be used by the
        training function in the Model class. 
        @layers - a list of layers fully describing a model.
        @output - the symbolic output of the network, see Model.compile()
        """
        pass

class SGD(Optimizer) :
    """
    Performs simple stochastic gradient descent on a network with a given learning rate,
    you can also add in a momentum parameter should you feel so inclined.
    """
    def __init__(self, lr=0.25, rho=0) :
        super(SGD, self).__init__(self)
        self.learn_rate = lr
        self.momentum = rho


    def updates(self, layers, output, costfn) :
        updates = []
        for ind, layer in enumerate(layers[::-1]) :
            updates.append((layer.weights, self.sgd_step(layer.weights, costfn)))
            updates.append((layer.bias, self.sgd_step(layer.bias, costfn)))
        return updates

    def sgd_step(self, weights, costfn) :
        """ 
        perform a single sgd update on the given weights using their gradient with respect
        to the error
        @TODO implement momentum yo
        """
        return weights - (self.learn_rate*T.grad(costfn, weights))
