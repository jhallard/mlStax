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

enum class Initializer {
    uniform,
    normal, 
    zero
};
enum class Activation {
    tanh,
    sigmoid,
    relu
}


// base class for the Layer hierarchy
class Layer {
    
public :

    virtual Layer(int layer_size, int input_dim, 
                    Initializer init = Initializer::normal, 
                    Activation act = Activation::sigmoid);

    virtual bool feed(Eigen::Vector2d data) = 0;

    virtual bool bprop(Eigen::MatrixXf error, bool verbose=false) = 0;

private :
    int input_dim;
    int layer_size; 
    std::string lname;
    Initializer init;
    Activations activation;

}

#endif
