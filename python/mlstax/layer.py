"""
author : John Allard (github: jhallard)
date : 2/14/2016
LICENSE : MIT
"""

import numpy as np


class Layer :
    """
    Base class for the Layer hierarchy. Each class represents a single
    layer in a deep neural network. The layers are then chained together under
    a model class along with activation functions.
    """
    def __init__(self, size, input_dim, init='uniform', activation=None) :
        self.activation = activation
        self.indim = input_dim
        self.init = init  # initialization function for weights
        self.size = size  # number of nodes in layer
        self.Wh = None    # weight matrix
        self.dWh = None   # derivative of above
        self.bh = None     # bias vector
        self.hs = None    # hidden state
        self.dhs = None   # d/dx matrix for hidden state

    def feed(self, data) :
        """
        feed the data from the previous layer through this one.
        """
        pass
    
    def bprop(self, dWback) :
        """
        takes the gradient data from the previous layer and uses it to update 
        the weights of this layer. Returns the gradient info for this layer.
        """
        pass

class Dense(Layer) :
    """
    Represents a layer that has a fully dense connection with the previous layer
    in the network (all inputs connect to all outputs)
    """
    def __init__(self, size, input_dim, *args, **kwargs) :
        super(Layer, self).__init__(args, kwargs)

        self.Wh = np.zeros(self.input_dim, self.size)
        self.hs = np.zeros(self.size, 1)
        self.b  = np.zeros(self.size, 1)

    def feed(self, data) :
        self.hs = np.dot(self.Wh, data) + self.bh
        pass

