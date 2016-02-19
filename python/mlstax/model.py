"""
author : John Allard (github: jhallard)
date : 2/14/2016
LICENSE : MIT
"""

import numpy as np

class Model :
    """
    Master model class. Stores a customized hierarchy of layers representing the
    DNN. Once built, can be trained by calling train(), and used for prediction by
    calling predict()
    """
    def __init__(input_dim, layers=[])
        self.indim = indim
        self.layers = layers

    def push_layer(layer) :
        """
        Add a new hidden layer to the network. 
        input must be a valid instance of a class in the Layer
        hierarchy
        """
        self.layers.push(layer)

    def compile(optimizer) :
        """
        Takes all of the layers and connects them together, prepares the network
        to be run. We need a valid instance of the optimizer class.
        """
        return

    def train(data, targets, nepochs) :
        """
        data - numpy style matrices of data to train on.
        nepochs - number of epochs to train for
        optimizer - instance of an Optimizer class (SGD, etc)
        """
        return

    def predict(data) :
        return

    def load_model(fn) :
        return 

    def load_weights(fn) :
        return
