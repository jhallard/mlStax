/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions for the various
 * initializer classes that are included with the project.
 */

#include "initializers.h"

using namespace mlstax;

// base class function defintions
void Initializer::set_name(std::string name) {
    m_name = name;
}

std::string Initializer::get_name() const {
    return m_name;
}

// Normal distribution function defintions
Normal::Normal(double mean, double stddev) : m_mean(mean), m_stddev(stddev), m_eng(m_rd()) {
    set_name("Normal");
}

void Normal::init_weights(Eigen::MatrixXd * inmat) {

}

// Uniform distribution function defintions
Uniform::Uniform(double min, double max) : m_min(min), m_max(max), m_eng(m_rd()) { 
    set_name("Uniform");
}

void Uniform::init_weights(Eigen::MatrixXd * inmat) {

}


// Constant distribution function definitions
Constant::Constant(double const_val) : m_const_val(const_val) {
    set_name("Constant");
}

void Constant::init_weights(Eigen::MatrixXd * inmat) {

}
