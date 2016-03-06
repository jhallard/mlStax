/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the declaration of the Dense Layer class.
 * This class is a member of the Layer hierarchy, and it implements a dense
 * (i.e. fully connected) connection of weights between the nodes at this layer
 * and the output of the nodes at the previous layer. It represent the 'standard'
 * layer in a feed-forward neural network.
 * */
#ifndef MLSTAX_LAYER_H
#define MLSTAX_LAYER_H


#include <vector>
#include <map>
#include <iostream>

#include <Eigen/Dense>

#include "layer.h"
#include "initializers.h"
#include "activations.h"

namespace mlstax {

// base class for the Layer hierarchy
class Dense : public Layer {

public :

    /*
    * @fn   : constructor
    * @args : layer_size - number of nodes in this layer, input_dim - output size of the last layer.
              init - instance of the Initializer hierarchy, used to initialize the weights for this layer
              act  - Activation function to be applied to the output of this layer.
    */
    virtual Dense(int layer_size, int input_dim, Initializer init = Normal(), Activation act = Sigmoid());

    /*
    * @fn   : feed
    * @args : `data` - input vector of data to this layer.
    * @ret  : Eigen::Vector2d, the output of this layer's transformation
    * @desc : feeds a vector of data through this layer and returns the output of the transformation
    */
    virtual bool feed(Eigen::Vector2d data);

    /*
    * @fn   : bprop
    * @args : error - The 'delta` from the next layer. verbose - if true we spit more info to stdout
    * @ret  : Eigen::MatrixXd, the error-delta associated with this layer for use with the next bprop call.
    * @desc : Takes an error from the next layer and computes it's gradient, then returns it's delta for the
    *         previous layer to make use of 
    */
    virtual bool bprop(Eigen::MatrixXd error, bool verbose=false);
    
    /*
    * @fn   : update
    * @args : none
    * @ret  : nothing (always true for now)
    * @desc : uses the internally saved gradient information and performs the relevant update step.
    */
    virtual bool update();

private :

};

#endif
