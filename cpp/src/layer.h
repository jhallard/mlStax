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
                    Activation act = Activation::sigmoid)

    virtual bool feed(Eigen::Vector2d data) = 0;

private :
    int input_dim;
    int layer_size; 
    std::string lname;
    Initializer init;
    Activations activation;

}

#endif
