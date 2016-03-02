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
    def __init__(self, input_dim, layers=[]) :
        self.indim = input_dim
        self.inputs = T.vector('inputs')
        self.outputs = T.vector('outputs')
        self.feed_forward = None # defined on call to compile()
        self.costfn = None # symbolic cost function for the network
        self.layers = []
        for layer in layers :
            self.push_layer(layer)

    def push_layer(self, layer) :
        """
        Add a new hidden layer to the network. 
        input must be a valid instance of a class in the Layer hierarchy
        """
        if self.layers :
            layer.init_weights(indim=self.layers[-1].size)
        else :
            layer.init_weights(indim=self.indim)
        layer.lnum = len(self.layers)+1
        self.layers.append(layer)

    def compile(self, optimizer=optimizer.SGD, costfn="MSE") :
        """
        Takes all of the layers and connects them together, prepares the network
        to be run. 
        @optimizer - instance of the optimizer class hierarchy
        """
        # symbolically run through the entire network
        temp_x = self.inputs
        for layer in layers :
            temp_x = layer.feed(output)

        # self.feed_forward hold the symbolic result for an output given an input
        self.feed_forward = temp_x
        self.costfn = self.get_costfn(costfn)
        
        # define a symbolic training iteration based on the input and output data,
        # the cost function, and the update algorithm defined in the optimizer class
        self._train = theano.function(
            inputs=[self.inputs, self.outputs],
            outputs=self.costfn,
            updates=optimizer.updates(self.layers, self.feed_forward, self.costfn)
        )

    def train(self, data, targets, batchsize=10, nepochs=10, verbose=False) :
        """
        data - numpy style input data
        targets - numpy style label data
        batchsize - iterations done before weight update
        nepochs - number of epochs to train for
        """
        pass    

    def predict(self, data) :
        pass

    def get_costfn(self, costfn) :
        if costfn == "MSE" :
            err = self.feed_forward - self.outputs
            return T.dot(err, err.T)

    def load_model(self, fn) :
        pass 

    def load_weights(self, fn) :
        pass

    def save_model(self, fn) :
        pass

    def save_weights(self, fn) :
        pass

    def __str__(self) :
        retstr = "Model Architecture :\n" 
        for i, layer in enumerate(self.layers) :
            retstr += str(layer) + "\n"
        return retstr
