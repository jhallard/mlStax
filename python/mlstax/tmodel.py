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
        self.input = T.vector('input')
        self.output = T.vector('output')
        self.feed_forward = None # defined on call to compile()
        for layer in layers :
            self.push_layer(layer)

    def push_layer(self, layer) :
        """
        Add a new hidden layer to the network. 
        input must be a valid instance of a class in the Layer hierarchy
        """
        if self.layers and self.layers[-1].size != layer.input_dim:
            raise ValueError("Layer Sizing Mismatch.\n Layers : %s --> %s" %
                        (str(self.layers[-1]), str(layer))
                   )
        layer.init_weights(indim=self.layers[-1].size)
        layer.lnum = len(self.layers)+1
        self.layers.append(layer)

    def compile(self, optimizer=optimizer.SGD) :
        """
        Takes all of the layers and connects them together, prepares the network
        to be run. 
        @optimizer - instance of the optimizer class hierarchy
        """
        temp_x = self.input
        for layer in layers :
            temp_x = layer.feed(output)

        # self.feed_forward hold the symbolic result for an output given an input
        self.feed_forward = temp_x
        self.cost = optimizer.costfn(self.feed_forward)
        
        # define a symbolic training iteration based on the input and output data,
        # the cost function, and the update algorithm defined in the optimizer class
        self._train = theano.function(
            inputs=[self.inputs, self.outputs],
            outputs=self.cost,
            updates=optimizer.updates(self.layers)
        )

    def train(self, data, targets, batchsize=10, nepochs=10, verbose=False) :
        """
        data - numpy style input data
        targets - numpy style label data
        batchsize - iterations done before weight update
        nepochs - number of epochs to train for
        """
        for epoch in range(nepochs) :
            toterr = 0
            for ind, datum in enumerate(data) :
                output = np.array([datum]).T
                for i, layer in enumerate(self.layers) :
                    output = layer.feed(output)
                error = output - targets[ind]
                loss = 0.5*(np.array(error).dot(np.array(error).T)[0][0])
                toterr += abs(loss)

                for i, layer in enumerate(self.layers[::-1]) :
                    error = layer.bprop(error)

                if ind != 0 and ind % batchsize == 0 :
                    for layer in self.layers :
                        layer.update()

            print "Loss for Epoch %s : %.6f" % (epoch, toterr/len(data))
                

    def predict(self, data) :
        retval = []
        for ind, datum in enumerate(data) :
            output = np.array([datum]).T
            for i, layer in enumerate(self.layers) :
                output = layer.feed(output)
            retval.append(output)
        return retval

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
