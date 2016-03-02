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

    def __init__(self, costfn="MSE") :
        self.costfn_str = costfn
        self.costfn = None # determined on compile from the costfn_str

    def compile(self, output, labels) :
        """
        compiles a loss function for use in the update step. takes a symbolic
        expression for the network output and the theano output shared variable
        and construct a symbolic expression for the loss function
        @output - the Model.feed_forward variable, i.e. the symbolic output of the net
        @labels - the net target-output shared variable i.e Model.outputs
        """
        pass

    def updates(self, layers) :
        """
        returns a list of theano shared variable updates to be used by the
        training function in the Model class. 
        @layers - a list of layers fully describing a model.
        """
        pass

class SGD(Optimizer) :
    """
    Performs simple stochastic gradient descent on a network with a given learning rate,
    you can also add in a momentum parameter should you feel so inclined.
    """
    def __init__(self, costfn="MSE", lr=0.25, rho=0) :
        super(SGD, self).__init__(costfn)
        self.learn_rate = lr
        self.momentum = rho

    def compile(self, outputs, labels) :
        if self.costfn_str == "MSE" :
            err = outputs - labels
            self.costfn = T.dot(err, err.T)[0][0]
            return self.costfn
        return None


    def updates(self, layers) :
        updates = []
        # for each layer in reverse, add the update expression for that layer's weights and bias
        for ind, layer in enumerate(layers[::-1]) :
            updates.append((layer.weights, self.sgd_step(layer.weights)))
            updates.append((layer.bias, self.sgd_step(layer.bias)))
        return updates

    def sgd_step(self, weights) :
        """ 
        perform a single sgd update on the given weights using their gradient with respect
        to the error
        @TODO implement momentum yo
        """
        return weights - (self.learn_rate*T.grad(self.costfn, weights))
