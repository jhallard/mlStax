/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the declaration of the base Layer class. This
 *  is an abstract base class that must be derived from to implement a legitimate
 *  network to be used by the Model class. Examples of child classes are the Dense
 *  and RNN layers defined in their respective files.
 * */
#ifndef MLSTAX_LAYER_H
#define MLSTAX_LAYER_H


#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>

#include <Eigen/Dense>

#include "initializers.h"
#include "activations.h"

namespace mlstax {

// @class : Layer
// @info  : base class for the Layer hierarchy
class Layer {
    
public :

    // overload the ostream operator to neatly print a description of the layer
    // that is compatible with std streams
    friend std::ostream& operator<<(std::ostream& os, const Layer& layer);

    /*
    * @fn   : constructor
    * @args : layer_size - number of nodes in this layer, input_dim - output size of the last layer.
    *         init - instance of the Initializer hierarchy, used to initialize the weights for this layer
    *         act  - Activation function to be applied to the output of this layer.
    */
    Layer(uint layer_size, uint input_dim, std::shared_ptr<Initializer> init, std::shared_ptr<Activation> act);

    /*
    * @fn   : feed
    * @args : `data` - input vector of data to this layer.
    * @ret  : Eigen::VectorXd, the output of this layer's transformation
    * @desc : feeds a vector of data through this layer and returns the output of the transformation
    */
    virtual std::shared_ptr<Eigen::VectorXd> feed(std::shared_ptr<Eigen::VectorXd> data) = 0;

    /*
    * @fn   : bprop
    * @args : error - The 'delta` from the next layer. verbose - if true we spit more info to stdout
    * @ret  : Eigen::MatrixXd, the error-delta associated with this layer for use with the next bprop call.
    * @desc : Takes an error from the next layer and computes it's gradient, then returns it's delta for the
    *         previous layer to make use of 
    */
    virtual std::shared_ptr<Eigen::MatrixXd> bprop(std::shared_ptr<Eigen::MatrixXd> error, bool verbose=false) = 0;
    
    /*
    * @fn   : update
    * @args : none
    * @ret  : nothing (always true for now)
    * @desc : uses the internally saved gradient information and performs the relevant update step.
    */
    virtual bool update() = 0;


    uint get_input_dim() const;
    uint get_layer_size() const;

    bool set_input_dim(uint indim);
    bool set_layer_size(uint lsize);

    // see http://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    // for explanation, tl;dr we need this for alignment so Eigen can use SIMD, w/o 
    // this we can get segfault on Layer * x = new Layer();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected :

    // Initializer and Activation function-objects for this specific layer
    std::shared_ptr<Initializer> m_initializer;
    std::shared_ptr<Activation> m_activation;

    uint m_input_dim, m_layer_size; 
    std::string m_name;
};

} // end namespace mlstax

#endif
