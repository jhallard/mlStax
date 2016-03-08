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
#include <memory>

#include <Eigen/Dense>

namespace mlstax {

class Activation {
public:
    // @func : activate
    // @args : a vector of doubles that you execute an element-wise transform over
    virtual void activate(Eigen::VectorXd & inmat) = 0;

    // @func : deactivate
    // @args : the output of the activate function that you want the derivative w/ 
    // respect to.
    virtual void dactivate(Eigen::VectorXd & activation) = 0;

	void set_name(std::string name);
	std::string get_name() const;
private:
	std::string m_name; // simply name the activation, like "Sigmoid" or w/e. Helps with debugging.
};

// @class : Sigmoid
// @info  : Performs the logistic sigmoid function elementwise over a given
// Eigen VectorXd. 
class Sigmoid : public Activation {
public:
    Sigmoid() { set_name("Sigmoid"); }
    virtual void activate(Eigen::VectorXd & inmat);
    virtual void dactivate(Eigen::VectorXd & inmat);
};

// @class :relu 
// @info  : performs the rectified linear-unit function elementwise over 
// a given eigen matrixxd. 
class ReLU : public Activation {
public:
    ReLU() { set_name("ReLU"); }
    virtual void activate(Eigen::VectorXd & inmat);
    virtual void dactivate(Eigen::VectorXd & inmat);
};

// @class : Tanh
// @info  : performs the hyperbolic tangent function elementwise over
// a given eigen matrixxd. 
class Tanh : public Activation {
public:
    Tanh() { set_name("Tanh"); }
    virtual void activate(Eigen::VectorXd & inmat);
    virtual void dactivate(Eigen::VectorXd & inmat);
};

// @class : Nothing 
// @info  : Performs no transformation on the data, can sometimes be convenient
class Nothing : public Activation {
public:
    Nothing() { set_name("Nothing"); }
    virtual void activate(Eigen::VectorXd & inmat);
    virtual void dactivate(Eigen::VectorXd & inmat);
};

} // endnamespace

#endif
