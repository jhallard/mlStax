/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions of the member functions for the
 * Dense class, the base class in the Dense heirarchy.
 */

#include "dense.h"

namespace mlstax {

Dense::Dense(uint layer_size, uint input_dim, Initializer * init, Activation * act) 
    : Layer(layer_size, input_dim, init, act), m_last_input(nullptr)
{
  m_name = "Dense";
  m_weights = Eigen::MatrixXd(m_layer_size, m_input_dim);
  m_dweights = Eigen::MatrixXd(m_layer_size, m_input_dim);
  m_bias = Eigen::VectorXd(m_layer_size, 1);
  m_dbias = Eigen::VectorXd(m_layer_size, 1);

  m_last_input = Eigen::VectorXd(m_layer_size, 1);
  m_hidden_state = Eigen::VectorXd(m_layer_size, 1);
}

bool Dense::feed(std::shared_ptr<Eigen::VectorXd> indat) {
    return true;
}

bool Dense::bprop(std::shared_ptr<Eigen::MatrixXd> error, bool verbose) {
    return true;
}

bool Dense::update() {
    return true;
}

} // end namespace
