/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions of the member functions for the
 * Dense class, the base class in the Dense heirarchy.
 */

#include "dense.h"

namespace mlstax {

Dense::Dense(uint layer_size, uint input_dim, std::shared_ptr<Initializer> init, std::shared_ptr<Activation> act)
    : Layer(layer_size, input_dim, init, act)
{
    m_name = "Dense";
    m_weights = Eigen::MatrixXd(m_layer_size, m_input_dim);
    m_initializer->init_weights(m_weights);
    m_dweights = Eigen::MatrixXd::Zero(m_layer_size, m_input_dim);
    m_bias = Eigen::VectorXd::Zero(m_layer_size, 1);
    m_dbias = Eigen::VectorXd::Zero(m_layer_size, 1);

    m_last_input = Eigen::VectorXd(m_layer_size, 1);
    m_hidden_state = Eigen::VectorXd::Zero(m_layer_size, 1);
    m_dhidden_state = Eigen::VectorXd::Zero(m_layer_size, 1);
}

std::shared_ptr<Eigen::VectorXd> Dense::feed(std::shared_ptr<Eigen::VectorXd> indat) {
    m_last_input = *indat;
    m_hidden_state = m_weights*(*indat) + m_bias;
    m_activation->activate(m_hidden_state);
    return std::make_shared<Eigen::VectorXd>(m_hidden_state);
}

std::shared_ptr<Eigen::MatrixXd> Dense::bprop(std::shared_ptr<Eigen::MatrixXd> error, bool verbose) {
    m_dhidden_state = m_hidden_state;
    m_activation->dactivate(m_dhidden_state);
    auto newdelta = error->cwiseProduct(m_dhidden_state);
    auto newerror = m_weights.transpose()*newdelta;
    m_dweights += newdelta*m_last_input.transpose();
    m_dbias += newdelta;
    return std::make_shared<Eigen::MatrixXd>(newerror);
}

bool Dense::update() {
    // std::cout << "dweights : " << m_dweights << std::endl; </br> </br>
    m_weights -= 0.10*m_dweights;
    m_bias -= 0.10*m_dbias;
    m_dweights = Eigen::MatrixXd::Zero(m_layer_size, m_input_dim);
    m_dbias = Eigen::VectorXd::Zero(m_layer_size, 1);
    // std::cout << "weights" << m_weights << std::endl;
    // std::cout << "bias : " << m_bias << std::endl;
    return true;
}

} // end namespace
