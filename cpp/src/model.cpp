/**
 * Author  - John Allard
 * Date    - Feb 27th 2016
 * License - MIT
 * Description - This file contains the declaration of the Model class.
 *  This class defines a standard deep neural network container that wraps around
 *  an ordered list of Layer objects. Input is then fed to the model allowing it to
 *  learn from the input and adjust itthis->accordingly.
 * */
#include "model.h"

namespace mlstax {

Model::Model(uint input_dim, std::vector<Layer*> layers) :
    m_input_dim(input_dim),
    m_layers(layers) 
{}

bool Model::push_layer(Layer * layer) {

    if(!m_layers.size()) {
        if(layer->get_input_dim() != this->m_input_dim) {
            // @TODO raise proper sizing error message / exception
            return false;
        }
    }
    else {
        if(layer->get_input_dim() != this->m_layers.back()->get_layer_size()) {
            // @TODO raise proper sizing error message / exception
            return false;
        }
    }
        this->m_layers.push_back(layer);
        return true;
}

std::vector<Layer*> Model::get_layers() const {
    return this->m_layers;
}

std::vector<EpochResult> Model::train(std::vector<Eigen::Vector2d> & indat,
        std::vector<Eigen::Vector2d> & targets,
        uint batchsize, uint nepochs, bool verbose) {

    std::vector<EpochResult> results = {};

    // perform training

    return results;
}

std::vector<EpochResult> Model::evaluate(std::vector<Eigen::Vector2d> & indat,
        std::vector<Eigen::Vector2d> & targets, bool verbose) {

    std::vector<EpochResult> results = {};

    // perform evaluation

    return results;

}

std::ostream& operator<<(std::ostream& os, const Model& model) {
    std::stringstream ss;
    ss << "Model Architecture : \n";
    for(int i = 0; i < model.m_layers.size(); i++) {
        ss << "Layer " << i << " : " << *(model.m_layers[i]) << "\n";
    } 
    os << ss.str();
    return os;
}


}
