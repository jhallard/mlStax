"""
author : John Allard (github: jhallard)
MLstax - A simple Deep-Learning framework modeled after Keras.
Build models by stacking layers, activation functions, and other 'blocks'
on-top of one another. Train and use the models for prediction purposes.

Made as part of my final project for CS112 at UCSC, Winter 2016
LICENSE : DWTFYW Public License
"""

import numpy as np

class Model :
    """
    Master model class. Stores a customized hierarchy of layers representing the
    DNN. Once built, can be trained by calling train(), and used for prediction by
    calling predict()
    """
    def __init__(input_dim)
        self.indim = indim

    def train(data, targets, nepochs, optimizer) :
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
        

class Layer :
    """
    Base class for the Layer hierarchy. Each class represents a single
    layer in a deep neural network. The layers are then chained together under
    a model class along with activation functions.
    """
    def __init__(dim) :
        self.dim = dim

