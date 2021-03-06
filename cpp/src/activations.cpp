/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the definitions for all of the
 * standard Activation classes that are included with the library.
 */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

#include "activations.h"

using namespace mlstax;

double sigmoid(double x) {
    return 1.0 / (1 + exp(x));
}

double my_tanh(double x) {
    return tanh(x);
}

double relu(double x) {
    if(x > 0) return x;
    else return 0;
}

void Activation::set_name(std::string name) {
    m_name = name;
}

std::string Activation::get_name() const {
    return m_name;
}

void Sigmoid::activate(Eigen::VectorXd & inmat) {
    inmat = inmat.unaryExpr(&sigmoid);
}

void Sigmoid::dactivate(Eigen::VectorXd & inmat) {
    inmat = inmat.unaryExpr(
            std::function<double(double)>([this](double sig) -> double { return sig*(1-sig); })
    );
}

void ReLU::activate(Eigen::VectorXd & inmat) {
    inmat = inmat.unaryExpr(&relu);
}

void ReLU::dactivate(Eigen::VectorXd & inmat) {
    inmat = inmat.unaryExpr(
            std::function<double(double)>([this](double sig) -> double { if(sig > 0) return sig; else return 0; })
    );
}

void Tanh::activate(Eigen::VectorXd & inmat) {
    inmat = inmat.unaryExpr(&my_tanh);
}

void Tanh::dactivate(Eigen::VectorXd & inmat) {
    inmat = inmat.unaryExpr(
            std::function<double(double)>([this](double sig) -> double { return 1.0 - tan(sig)*tan(sig); })
    );
}

void Nothing::activate(Eigen::VectorXd & inmat) {
    //literally do nothing, leave inmat alone.
}

void Nothing::dactivate(Eigen::VectorXd & inmat) {
    // f(x) = x, f'(x) = 1
    inmat = inmat.unaryExpr(
            std::function<double(double)>([this](double sig) -> double { return 1.0; })
    );
}

