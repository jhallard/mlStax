/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions of the member functions for the
 * Dense class, the base class in the Dense heirarchy.
 */

#include "dense.h"

Dense::Dense(uint layer_size, uint input_dim, Initializer * init, Activation * act) 
    : Layer(layer_size, input_dim, init, act)
{}

bool Dense::feed(Eigen::Vector2d * indat) {
  return true;
}

bool Dense::bprop(Eigen::MatrixXd * error, verbose) {
    return true;
}

