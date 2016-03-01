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
        self.layers = layers

    def push_layer(self, layer) :
        """
        Add a new hidden layer to the network. 
        input must be a valid instance of a class in the Layer
        hierarchy
        """
        if self.layers and self.layers[-1].size != layer.input_dim:
            raise ValueError("Layer Sizing Mismatch.\n Layers : %s --> %s" %
                        (str(self.layers[-1]), str(layer))
                   )
        layer.lnum = len(self.layers)+1
        self.layers.append(layer)

    def compile(self, optimizer) :
        """
        Takes all of the layers and connects them together, prepares the network
        to be run. We need a valid instance of the optimizer class.
        """
        pass

    def train(self, data, targets, batchsize=10, nepochs=10, verbose=False) :
        """
        data - numpy style matrices of data to train on.
        nepochs - number of epochs to train for
        optimizer - instance of an Optimizer class (SGD, etc)
        """
        for epoch in range(nepochs) :
            toterr = 0
            for ind, datum in enumerate(data) :
                output = np.array([datum]).T
                for i, layer in enumerate(self.layers) :
                    output = layer.feed(output)
                # if ind % 33 == 0 :
                    # print "%s, %s" % (targets[ind], output)
                error = output - targets[ind]
                # print "error : %s" % error
                loss = 0.5*(np.array(error).dot(np.array(error).T))
                toterr += abs(loss)

                for i, layer in enumerate(self.layers[::-1]) :
                    error = layer.bprop(error)

                for layer in self.layers :
                    layer.update()

            print "Loss for Epoch %s : %s" % (epoch, toterr/len(data))
                

    def predict(self, data) :
        retval = []
        for ind, datum in enumerate(data) :
            output = np.array([datum]).T
            for i, layer in enumerate(self.layers) :
                output = layer.feed(output)
            print output
            retval.append(output)
        return retval

    def load_model(self, fn) :
        pass 

    def load_weights(self, fn) :
        pass

    def __str__(self) :
        xx= "Model Architecture :\n" 
        for i, layer in enumerate(self.layers) :
            xx += str(layer) + "\n"
        return xx
