/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - These classes define all of the different activation
 * functions that can be used with the various layers of your network.
 * These activation functions are arranged in a hierarchy to allow the
 * exploitation of polymorphism.
 * */
#ifndef MLSTAX_ACTIVATION_H
#define MLSTAX_ACTIVATION_H

#include <vector>
#include <map>
#include <iostream>
#include <random>

#include <Eigen/Dense>

namespace mlstax {
using namespace mlstax;

class Activation {
public:
    virtual void activate(Eigen::MatrixXd *) = 0;
};

// @class : Sigmoid
// @info  : Performs the logistic sigmoid function elementwise over a given
// Eigen MatrixXd. 
class Sigmoid : public Activation {
public:
    Sigmoid();
    virtual void activate(Eigen::MatrixXd * inmat);
};

// @class :relu 
// @info  : performs the rectified linear-unit function elementwise over 
// a given eigen matrixxd. 
class ReLU : public Activation {
public:
    ReLU();
    virtual void activate(Eigen::MatrixXd * inmat);
};

// @class : Tanh
// @info  : performs the hyperbolic tangent function elementwise over
// a given eigen matrixxd. 
class Tanh : public Activation {
public:
    Tanh();
    virtual void activate(Eigen::MatrixXd * inmat);
};

} // endnamespace

#endif
