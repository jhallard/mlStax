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

#include "../src/model.h"
#include "../src/dense.h"
#include "../src/initializers.h"
#include "../src/activations.h"

using namespace mlstax;
using namespace std;

#define STR_DIV cout << "_________________________________________________" << endl;

// @test : model_construct
// @info : simply construct a model with no layers and return true if nothing breaks
bool model_construct() {
    try {
        Model mm = Model(12);
        cout << "Model Created. Architecture Below" << endl;
        STR_DIV
        cout << mm;
        STR_DIV
        STR_DIV
    } catch(...) {
        cerr << "Model Construction Failed (model_construct)" << endl;
        return false;
    }
    return true;
}

// @test : add_layers
// @info : construct a model and add a single dense layer to it. Success if nothing breaks
bool add_layers() {
    try {
        Model mm = Model(12);
        Initializer * init = new Normal(0, 2.0);
        Activation * act = new Sigmoid();
        Dense * layer = new Dense(20, 12, init, act); // use default inti and activations
        mm.push_layer(layer);
        cout << "Model Created and Layer Added" << endl;
        STR_DIV
        cout << mm;
        STR_DIV
        STR_DIV
        return true;
    } catch(...) {
        cerr << "Adding Dense Layer to Model Failed (add_layers)" << endl;
        return false;
    }
}


// @test : test_initializers
// @info : mostly for visual confirmation that stuff is being randomized, we aren't doing
// any statistical checks on the randomness.
bool test_initializers() {
    Initializer * init = new Uniform(-1, 1);
    Eigen::MatrixXd inmat = Eigen::MatrixXd::Zero(2, 2);
    init->init_weights(inmat);
    delete(init);
    cout << inmat << endl;

    init = new Normal(50, 20);
    inmat = Eigen::MatrixXd::Zero(2, 2);
    init->init_weights(inmat);
    delete(init);
    cout << inmat << endl;

    init = new Constant(44.0);
    inmat = Eigen::MatrixXd::Zero(2, 2);
    init->init_weights(inmat);
    delete(init);
    cout << inmat << endl;
    
    return true;
}

bool test_activations() {
    // confirmed as working
    Activation * act = new Sigmoid();
    Eigen::VectorXd indat = Eigen::VectorXd::Random(10);
    act->activate(indat);
    delete(act);

    // confirmed as working
    act = new ReLU();
    indat = Eigen::VectorXd::Random(10);
    act->activate(indat);
    delete(act);

    act = new Tanh();
    indat = Eigen::VectorXd::Random(10);
    cout << indat << endl << endl;
    act->activate(indat);
    delete(act);
    cout << indat << endl;

    return true;
}

int main(int argc, char **argv) {
    cout << "Starting Unit Tests :" << endl;
    test_initializers() && test_activations();
    return model_construct() && add_layers();
}
