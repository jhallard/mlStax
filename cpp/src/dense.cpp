/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions of the member functions for the
 * Dense class, the base class in the Dense heirarchy.
 */

#include "dense.h"

using namespace mlstax;

Dense::Dense(uint layer_size, uint input_dim, Initializer * init, Activation * act) 
    : Layer(layer_size, input_dim, init, act)
{
  m_name = "Dense";
}

bool Dense::feed(std::shared_ptr<Eigen::Vector2d> indat) {
    return true;
}

bool Dense::bprop(std::shared_ptr<Eigen::MatrixXd> error, bool verbose) {
    return true;
}

bool Dense::update() {
    return true;
}

