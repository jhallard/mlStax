#ifndef MLSTAX_LAYER_H
#define MLSTAX_LAYER_H


#include <vector>
#include <map>

class Layer {
    
public :
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

    Layer(int layer_size, int input_dim, 
          Initializer init = Initializer::normal,
          Activation act = Activation::sigmoid)

private :
    int input_dim;
    int layer_size; 
    std::string lname;
    Initializer init;
    Activations activation;


}





#endif
