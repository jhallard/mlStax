 /**                                                                                                                               
   * Author  - John Allard
   * Date    - Feb 27th 2016
   * License - MIT
   * Description - This file contains a series of simple unit tests for the C++ mlstax
   * library. These tests will build, train, evaluate simple models just to give us a sanity
   * check to rely on while building.
   * */
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "model.h"
#include "dense.h"
#include "initializers.h"
#include "activations.h"

using namespace mlstax;
using namespace std;
// @test : model_construct
// @info : simply construct a model with no layers and return true if nothing breaks
bool model_construct() {
    try {
        Model mm = Model(12);
    } catch(...) {
        std::cerr << "Model Construction Failed (model_construct)" << endl;
        return false;
    }
    return true;
}

// @test : add_layers
// @info : construct a model and add a single dense layer to it. Success if nothing breaks
bool add_layers() {
    try {
        Model mm = Model(12);
        Dense * layer = new Dense(20, 12); // use default inti and activations
        mm.push_layer(layer);
        return true;
    } catch(...) {
        std::cerr << "Adding Dense Layer to Model Failed (add_layers)" << endl;
        return false;
    }

int main(int *argv, char **argv) {
    return model_construct() && add_layers();
}
