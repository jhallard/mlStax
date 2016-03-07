/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - These classes define the different ways that one can
 * initialize the weights for a given layer. We define a base class that
 * all Layer object can accept upon construction, because of polymorphism 
 * you can make your own child classes of Initializer to customize the weight
 * initializtions for your network
 * */
#ifndef MLSTAX_INIT_H
#define MLSTAX_INIT_H

#include <vector>
#include <map>
#include <iostream>
#include <random>

#include <Eigen/Dense>

namespace mlstax {
using namespace mlstax;

// @class : Initializer
// @description : Abstract base class, all instances of this hierarchy can be used to
// initialize a given Eigen matrix to certiain values.
class Initializer {
public :
    virtual void init_weights(Eigen::MatrixXd * inmat) = 0;
};


// @class : Normal
// @description : Initialize an Eigen matrix with random values chosen from a
//   normal distribution with the given mean and standard deviation.
class Normal : public Initializer {
private :
    std::random_device rd;
    std::mt19937 eng;
public :
    Normal(double mean = 0, double stddev = 1.0);
    virtual void init_weights(Eigen::MatrixXd * inmat);
};

// @class : Uniform
// @description : Initialize a Eigen matrix with random values chosen from a uniform 
//   distribution. Distribution is given max and min values.
class Uniform : public Initializer {
private :
    std::random_device rd;
    std::mt19937 eng;
public :
    Uniform(double max = -1.0, double min = 1.0);
    virtual void init_weights(Eigen::MatrixXd * inmat);
};

// @class : Constant 
// @description : Initialize a Eigen matrix with constant values
class Constant : public Initializer {
private:
    float const_val;
public :
    Constant(float const_val);
    virtual void init_weights(Eigen::MatrixXd * inmat);
};

} // end namespace mlstax

#endif
