/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions for all of the
 * standard Activation classes that are included with the library.
 */

#include "activations.h"

using namespace mlstax;

void Activation::set_name(std::string name) {
    m_name = name;
}

std::string Activation::get_name() const {
    return m_name;
}

void Sigmoid::activate(Eigen::MatrixXd * inmat) {

}


void ReLU::activate(Eigen::MatrixXd * inmat) {

}

void Tanh::activate(Eigen::MatrixXd * inmat) {

}

void Nothing::activate(Eigen::MatrixXd * inmat) {

}
