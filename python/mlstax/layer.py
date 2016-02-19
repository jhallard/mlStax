"""
author : John Allard (github: jhallard)
date : 2/14/2016
LICENSE : MIT
"""


class Layer :
    """
    Base class for the Layer hierarchy. Each class represents a single
    layer in a deep neural network. The layers are then chained together under
    a model class along with activation functions.
    """
    def __init__(size, input_dim, init='uniform', activation=None) :
        self.size = size  # number of nodes in layer
        self.activation = activation
        self.Wh = None    # weight matrix
        self.dWh = None   # derivative of above
        self.indim = None # input dimension
        self.init = init  # initialization function for weights

    def feed(data) :
        """
        feed the data from the previous layer through this one.
        """
        return
    
    def bprop(dWback) :
        """
        takes the gradient data from the previous layer and uses it to update 
        the weights of this layer. Returns the gradient info for this layer.
        """

class Dense(Layer) :
    """
    Represents a layer that has a fully dense connection with the previous layer
    in the network (all inputs connect to all outputs)
    """
    def __init__(size, input_dim, *args, **kwargs) :
        super(Layer, self).__init__(args, kwargs)

        self.Wh = np.zeros(input_dim, size)

    def feed(data) :

