/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the declaration of the Model class.
 *  This class defines a standard deep neural network container that wraps around
 *  an ordered list of Layer objects. Input is then fed to the model allowing it to
 *  learn from the input and adjust itself accordingly.
 * */
#ifndef MLSTAX_MODEL_H
#define MLSTAX_MODEL_H

#include <vector>
#include <map>
#include <iostream>
#include <sstream>

#include <Eigen/Dense>

#include "layer.h"

namespace mlstax {

struct EpochResult {
    double loss;
    double accuracy;
};

class Model {

public :
    friend std::ostream& operator<<(std::ostream& os, const Model& model);

    Model(uint input_dim, std::vector<std::shared_ptr<Layer>> layers = {});

    bool push_layer(std::shared_ptr<Layer> layer);
    std::vector<std::shared_ptr<Layer>> get_layers() const;

    std::vector<EpochResult> train(std::vector<Eigen::VectorXd> & indat, 
           std::vector<Eigen::VectorXd> & targets,
           uint batchsize = 10, uint nepochs = 10, bool verbose=false
    ); 

    std::vector<EpochResult> evaluate(std::vector<Eigen::VectorXd> & indat, 
           std::vector<Eigen::VectorXd> & targets,
           bool verbose=false
    ); 

    std::vector<Eigen::VectorXd> predict(std::vector<Eigen::VectorXd> & indat);

    bool save_weights(const std::string fn) const;
    bool load_weights(const std::string fn);

    bool save_model(const std::string fn) const;
    bool load_model(const std::string fn);

private :
    uint m_input_dim;
    std::vector<std::shared_ptr<Layer>> m_layers;

};


}

#endif
