/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions of the member functions for the
 * Layer class, the base class in the Layer heirarchy.
 */

#include "layer.h"

using namespace mlstax;

Layer::Layer(uint layer_size, uint input_dim, Initializer * init, Activation * act) 
    : m_layer_size(layer_size), m_input_dim(input_dim),
      m_initializer(init),
      m_activation(act) 
{} 

uint Layer::get_input_dim() const {
    return m_input_dim;
}

uint Layer::get_layer_size() const {
    return m_layer_size;
}

bool Layer::set_input_dim(uint indim) {
    m_input_dim = indim;
}

bool Layer::set_layer_size(uint lsize) {
    m_layer_size = lsize;
}
