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

#include <Eigen/Dense>

namespace mlstax {

class Model {

public :
    Model(int input_dim, std::vector<Layer> layers = {});

    bool push_layer(Layer * layer);
    std::vector<Layer> get_layers() const;

    std::vector<std::string> train(std::vector<Eigen::Vector2d> & indat, 
           std::vector<Eigen::Vector2d> * targets,
           uint batchsize = 10, uint nepochs = 10, verbose=false
    ); 

    std::vector<std::string> evaluate(std::vector<Eigen::Vector2d> & indat, 
           std::vector<Eigen::Vector2d> * targets,
           verbose=false
    ); 

    std::vector<Eigen::Vector2d> predict(std::vector<Eigen::Vector2d> * indat);

    bool save_weights(const std::string fn) const;
    bool load_weights(const std::string fn);

    bool save_model(const std::string fn) const;
    bool load_model(const std::string fn);

private :

    int m_input_dim;
    std::vector<Layer*> m_layers;
}

}

#endif
