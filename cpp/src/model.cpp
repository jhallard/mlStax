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

Model::Model(uint input_dim, std::vector<std::shared_ptr<Layer>> layers) :
    m_input_dim(input_dim),
    m_layers(layers) 
{}

bool Model::push_layer(std::shared_ptr<Layer> layer) {

    if(!m_layers.size()) {
        if(layer->get_input_dim() != this->m_input_dim) {
            return false;
        }
    }
    else {
        if(layer->get_input_dim() != this->m_layers.back()->get_layer_size()) {
            return false;
        }
    }
        this->m_layers.push_back(layer);
        return true;
}

std::vector<std::shared_ptr<Layer>> Model::get_layers() const {
    return this->m_layers;
}

std::vector<EpochResult> Model::train(
        std::vector<Eigen::VectorXd> & indat,
        std::vector<Eigen::VectorXd> & targets,
        uint batchsize, uint nepochs, bool verbose) {

    std::vector<EpochResult> results = {};

    for(uint i = 0; i < nepochs; i++) {
        double totloss = 0.0;
        for(uint n = 0; n < indat.size(); n++) {
            auto throughdata = std::make_shared<Eigen::VectorXd>(indat[n]);
             // std::cout << "Feeding Forward...\n";
             // std::cout << "Input : \n" << *throughdata << "\n\n";
            for(auto &layer : m_layers) {
                throughdata = layer->feed(throughdata);
                // std::cout << "Intermediate : " << *throughdata << "\n\n";
            }
            std::cout << "Output : " << *throughdata << ", Target : " << targets[n] << std::endl;
            auto error = std::make_shared<Eigen::MatrixXd>(*throughdata - targets[n]);
            // std::cout << "Error : " << *error << std::endl;
            auto loss = (error->transpose())*(*error);
            std::cout << "Loss : " << loss << std::endl;
            totloss += loss(0);

            auto rit = m_layers.rbegin();
            for (; rit!= m_layers.rend(); ++rit) {
                error = (*rit)->bprop(error);
            }

            for(auto &layer : m_layers) {
                layer->update();
            }
        }

        std::cout << "Epoch " << i << " Finished, average loss = " << totloss/indat.size() << "\n\n";

    }

    return results;
}

std::vector<EpochResult> Model::evaluate(std::vector<Eigen::VectorXd> & indat,
        std::vector<Eigen::VectorXd> & targets, bool verbose) {

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
