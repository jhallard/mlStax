"""
author : John Allard (github: jhallard)
date : 2/14/2016
LICENSE : MIT
"""

import numpy as np

def sigmoid(x) :
    return 1.0 / (1.0 + np.exp(-x))

def relu(x) :
    return x * (x>0)


class Layer(object) :
    """
    Base class for the Layer hierarchy. Each class represents a single
    layer in a deep neural network. The layers are then chained together under
    a model class along with activation functions.
    """

    def __init__(self, size, input_dim, init='uniform', activation="") :
        self.input_dim = input_dim
        self.lname = ""   # children will overwrite
        self.init = init  # initialization function for weights
        self.size = size  # number of nodes in layer
        self.Wh = None    # weight matrix
        self.dWh = None   # derivative of above
        self.bh = None    # bias vector
        self.hs = None    # hidden state (post activation function)
        self.acc = None   # accumulation (pre activation function)
        self.dhs = None   # d/dx matrix for hidden state
        self.act_default = "sigmoid"
        self.actstr = activation if activation else self.act_default
        self.activations = {
                "tanh" : np.tanh,
                "sigmoid" : sigmoid,
                "relu" : relu
        }
        self.dactivations = {
                "tanh" : np.tanh,
                "sigmoid" : lambda x : x*(1-x),
                "relu" : lambda x : 1 * (x>0)
        }
        self.activation = self.activations.get(self.actstr, self.act_default)
        self.dactivation = self.dactivations.get(self.actstr, self.act_default)


    def feed(self, data) :
        """
        feed the data from the previous layer through this one.
        """
        pass
    
    def bprop(self, error) :
        """
        takes the gradient data from the previous layer and uses it to update 
        the weights of this layer. Returns the gradient info for this layer.
        """
        pass

    def __str__(self) :
        return """type : %s, indim : %s, init : %s, size : %s, activation : %s""" % \
              (self.lname, self.input_dim, self.init, self.size, self.activation)



class Dense(Layer) :
    """
    Represents a layer that has a fully dense connection with the previous layer
    in the network (all inputs connect to all outputs)
    """
    def __init__(self, size, input_dim, **kwargs) :
        super(Dense, self).__init__(size, input_dim, **kwargs)
        self.lname = "Dense"
        self.Wh = np.random.randn(self.input_dim, self.size)*0.02
        self.hs = np.random.randn(self.size, 1)*0.02
        self.bh  = np.random.randn(self.size, 1)*0.02

    def feed(self, data) :
        self.acc = np.dot(data, self.Wh)
        self.hs = self.activation(self.acc)
        # print self.acc
        # print self.hs
        return self.hs

    def bprop(self, err) :
        newdelta = err * self.dactivation(self.hs)
        newerr = newdelta.dot(self.Wh.T)
        self.Wh += self.hs.T.dot(err)
        return newerr
