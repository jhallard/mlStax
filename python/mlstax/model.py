"""
author : John Allard (github: jhallard)
date : 2/14/2016
LICENSE : MIT
"""

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import sys

sys.path.append("../")
import layer, optimizers, initializers, activations

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
        self.layers = []
        self.costfn = None # grabbed on optimizer compilitation (see compile())
        self._train = None
        self._evaluate = None
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

    def compile(self, optimizer=optimizers.SGD()) :
        """
        Takes all of the layers and connects them together, prepares the network
        to be run. 
        @optimizer - instance of the optimizer class hierarchy
        """
        # symbolically run through the entire network
        temp_x = self.inputs
        for layer in self.layers :
            temp_x = layer.feed(temp_x)

        # self.feed_forward hold the symbolic result for an output given an input
        self.feed_forward = temp_x

        self.costfn = optimizer.compile(self.feed_forward, self.outputs)
        updates = optimizer.updates(self.layers)

        # define a symbolic training iteration based on the input and output data,
        # the cost function, and the update algorithm defined in the optimizer class
        self._train = theano.function(
            inputs=[self.inputs, self.outputs],
            outputs=self.costfn,
            updates=updates,
            name="train"
        )

        self._evaluate = theano.function(
            inputs=[self.inputs, self.outputs],
            outputs=[self.costfn, self.feed_forward], # grab the cost and the raw output for 
            name="evaluate"                           # the evaluation steps 
        )

    def train(self, data, targets, batchsize=10, nepochs=10, verbose=False) :
        """
        data - numpy style input data
        targets - numpy style label data
        batchsize - iterations done before weight update
        nepochs - number of epochs to train for
        """
        for epoch in range(nepochs) :
            toterr = 0.0
            for ind, datum in enumerate(data) :
                loss = self._train(
                        datum.astype(np.float32),
                        targets[ind].astype(np.float32)
                )
                toterr += loss
                # print loss
            print "Loss for Epoch %s : %s " % (epoch, toterr/len(data))

        print "fuck"


    def predict(self, data) :
        pass

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
