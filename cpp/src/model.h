#ifndef MLSTAX_MODEL_H
#define MLSTAX_MODEL_H

#include <vector>
#include <map>
#include <iostream>

#include <Eigen/Dense>

class Model {

public :
    Model(int input_dim, std::vector<Layer> layers = {});

    bool push_layer(Layer layer);
    std::vector<Layer> get_layers() const;

    bool save_weights(const std::string fn) const;
    bool load_weights(const std::string fn);

    bool save_model(const std::string fn) const;
    bool load_model(const std::string fn);

private :

    int _input_dim;
    std::vector<Layer> _layers;
}

#endif
