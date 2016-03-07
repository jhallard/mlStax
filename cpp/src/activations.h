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
	void set_name(std::string name);
	std::string get_name() const;
    virtual void activate(std::shared_ptr<Eigen::VectorXd> inmar) = 0;
private:
	std::string m_name;
};

// @class : Sigmoid
// @info  : Performs the logistic sigmoid function elementwise over a given
// Eigen VectorXd. 
class Sigmoid : public Activation {
public:
    Sigmoid() { set_name("Sigmoid"); }
    virtual void activate(std::shared_ptr<Eigen::VectorXd> inmat);
};

// @class :relu 
// @info  : performs the rectified linear-unit function elementwise over 
// a given eigen matrixxd. 
class ReLU : public Activation {
public:
    ReLU() { set_name("ReLU"); }
    virtual void activate(std::shared_ptr<Eigen::VectorXd> inmat);
};

// @class : Tanh
// @info  : performs the hyperbolic tangent function elementwise over
// a given eigen matrixxd. 
class Tanh : public Activation {
public:
    Tanh() { set_name("Tanh"); }
    virtual void activate(std::shared_ptr<Eigen::VectorXd> inmat);
};

// @class : Nothing 
// @info  : Performs no transformation on the data, can sometimes be convenient
class Nothing : public Activation {
public:
    Nothing() { set_name("Nothing"); }
    virtual void activate(std::shared_ptr<Eigen::VectorXd> inmat);
};

} // endnamespace

#endif
