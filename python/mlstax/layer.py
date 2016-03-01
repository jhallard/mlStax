"""
author : John Allard (github: jhallard)
date : 2/14/2016
LICENSE : MIT
"""

import numpy as np
import time

class Layer(object) :
    """
    Base class for the Layer hierarchy. Each class represents a single
    layer in a deep neural network. The layers are then chained together under
    a model class along with activation functions.
    """

    def __init__(self, size, input_dim, init='uniform', activation="", learn_rate=0.05) :
        self.input_dim = input_dim
        self.lname = ""   # children will overwrite
        self.lnum = -1    
        self.init = init  # initialization function for weights
        self.size = size  # number of nodes in layer
        self.Wh = None    # weight matrix
        self.dWh = None   # derivative of above
        self.bh = None    # bias vector
        self.hs = None    # hidden state (post activation function)
        self.acc = None   # accumulation (pre activation function)
        self.dhs = None   # d/dx matrix for hidden state
        self.learn_rate = learn_rate
        self.act_default = "relu"
        self.actstr = activation if activation else self.act_default
        self.activations = {
                "none" : lambda x : x,
                "tanh" : np.tanh,
                "sigmoid" : lambda x :  1.0 / (1.0 + np.exp(-x)),
                "relu" : lambda x : x * (x>0)
        }
        # note that these give f'(x) given f(x), not x.
        # i.e. sigmoid'(x) = sigmoid(x)*(1-sigmoid(x)), this assumes you pass in sigmoid(x) as x
        self.dactivations = {
                "none" : lambda x : 1,
                "tanh" : lambda x : 1 - x**2,
                "sigmoid" : lambda x : x*(1-x),
                "relu" : lambda x : 1 * (x > 0)
        }
        self.activation = self.activations.get(self.actstr, self.act_default)
        self.dactivation = self.dactivations.get(self.actstr, self.act_default)


    def feed(self, data, verbose=False) :
        """
        feed the data from the previous layer through this one.
        """
        pass
    
    def bprop(self, error, verbose=False) :
        """
        takes the gradient data from the previous layer and uses it to update 
        the weights of this layer. Returns the gradient info for this layer.
        """
        pass

    def update(self, verbose=False) :
        """
        updates the weight matrix by some combination of the learning rate and gradient
        """
        pass

    def __str__(self) :
        return """Layer #%s, type : %s\n indim : %s, size : %s\n init : %s, activation : %s""" % \
              (self.lnum, self.lname, self.input_dim, self.size, self.init, self.actstr)



class Dense(Layer) :
    """
    Represents a layer that has a fully dense connection with the previous layer
    in the network (all inputs connect to all outputs)
    """
    def __init__(self, size, input_dim, **kwargs) :
        super(Dense, self).__init__(size, input_dim, **kwargs)
        self.lname = "Dense"
        self.Wh = np.random.randn(self.size, self.input_dim)*1.0
        self.dWh = np.zeros_like(self.Wh)
        self.hs = np.random.randn(self.size, 1)*1.0
        self.bh  = np.zeros_like(self.hs)
        self.dbh  = np.zeros_like(self.bh)

    def feed(self, data, verbose=False) :
        # print self.Wh
        # print self.bh
        self.acc = np.dot(self.Wh, data) + self.bh
        self.hs = self.activation(self.acc)
        # print "___ Feed Forward ___"
        # print "data shape : %s" % str(data.shape)
        # print "Wh shape  : %s" % str(self.Wh.shape)
        # print "dWh shape : %s" % str(self.dWh.shape)
        # print "acc shape : %s" % str(self.acc.shape)
        # print "hs shape :  %s" % str(self.hs.shape)
        # print "\n\n"
        return self.hs

    def bprop(self, err) :
        newdelta = np.multiply(err, self.dactivation(self.hs))
        newerr = self.Wh.T.dot(newdelta)
        self.dWh += self.hs.T.dot(newdelta)
        self.dbh += newdelta
        np.clip(self.dWh, -50, 50, out=self.dWh)
        np.clip(self.dbh, -50, 50, out=self.dbh)
        # np.clip(self.Wh, -10, 10, out=self.dWh)
        # print "hs : " + str(self.hs)
        # print "newdelta : " + str(newdelta)
        # print "dWh: " + str(self.dWh.shape)
        return newerr

    def update(self) :
        # print self.dWh
        # print "___ Update __"
        # print "Wh shape  : %s" % str(self.Wh.shape)
        # print "dWh shape : %s" % str(self.dWh.shape)
        # print "hs shape  : %s" % str(self.hs.shape)
        # print "\n\n"
        self.Wh -= self.learn_rate * self.dWh
        self.bh -= self.learn_rate * self.dbh
        self.dWh = np.zeros_like(self.Wh)
        self.dbh = np.zeros_like(self.bh)
