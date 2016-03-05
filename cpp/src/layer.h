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
#include <iostream>

#include <Eigen/Dense>

namespace mlstax {
// base class for the Layer hierarchy
class Layer {
    
public :

    virtual Layer(int layer_size, int input_dim, 
        Initializer init = Normal, 
        Activation act = Sigmoid()
    );

    virtual bool feed(Eigen::Vector2d data) = 0;

    virtual bool bprop(Eigen::MatrixXd error, bool verbose=false) = 0;

private :

    // Initializer and Activation function-objects for this specific layer
    std::unique_ptr<Initializer> * activation;
    std::unique_ptr<Activation> * activation;
    int input_dim, layer_size; 
    std::string lname;
}

} // end namespace mlstax

#endif
