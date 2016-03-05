/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the declaration of the Model class.
 *  This class defines a standard deep neural network container that wraps around
 *  an ordered list of Layer objects. Input is then fed to the model allowing it to
 *  learn from the input and adjust itself accordingly.
 * */
#include "model.h"

Model::Model(int input_dim) m_input_dim(input_dim) {
  m_layers = {};
}

bool Model::push_layer(Layer * layer) {

    if(!m_layers.size()) {
        if(layer->get_input_dim() != self.input_dim) {
            // @TODO raise proper sizing error message / exception
            return false;
        }
        self.m_layers.push_back(layer);
        return true;
    }
    else {
        if(layer->get_input_dim() != self.layers.back()->get_layer_size()) {
            // @TODO raise proper sizing error message / exception
            return false;
        }
        self.m_layers.push_back(layer);
        return true;
    }
}
