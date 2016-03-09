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
        auto init = std::make_shared<Normal>(0, 2.0);
        auto act = std::make_shared<Sigmoid>();
        auto layer = std::make_shared<Dense>(20, 12, init, act); // use default inti and activations
        mm.push_layer(layer);
        auto init2 = std::make_shared<Uniform>(-2.0, 2.0);
        auto layer2 = std::make_shared<Dense>(24, 20, init2, act); // use default inti and activations
        mm.push_layer(layer2);
        return true;
    } catch(...) {
        cerr << "Adding Dense Layer to Model Failed (add_layers)" << endl;
        return false;
    }
}


bool simple_xor_test() {
    try {
        Model mm = Model(2);
        auto init = std::make_shared<Normal>(0, 0.2);
        auto act = std::make_shared<ReLU>();
        auto layer = std::make_shared<Dense>(5, 2, init, act); // use default inti and activations
        mm.push_layer(layer);
        auto init2 = std::make_shared<Uniform>(-0.2, 0.2);
        auto act2 = std::make_shared<ReLU>();
        auto layer2 = std::make_shared<Dense>(1, 5, init2, act2); // use default inti and activations
        mm.push_layer(layer2);
        STR_DIV
        cout << mm << std::endl;
        STR_DIV
        STR_DIV
        vector<Eigen::VectorXd> indata; 
        Eigen::VectorXd temp(2);
        temp << 0, 0;
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 0, 1).finished();
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 1, 0).finished();
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 1, 1).finished();
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 1, 0).finished();
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 0, 1).finished();
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 1, 1).finished();
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 0, 1).finished();
        indata.push_back(temp);
        temp = (Eigen::VectorXd(2) << 0, 0).finished();
        indata.push_back(temp);


        vector<Eigen::VectorXd> targets;
        targets.push_back((Eigen::VectorXd(1) << 0).finished());
        targets.push_back((Eigen::VectorXd(1) << 1).finished());
        targets.push_back((Eigen::VectorXd(1) << 1).finished());
        targets.push_back((Eigen::VectorXd(1) << 0).finished());
        targets.push_back((Eigen::VectorXd(1) << 1).finished());
        targets.push_back((Eigen::VectorXd(1) << 1).finished());
        targets.push_back((Eigen::VectorXd(1) << 0).finished());
        targets.push_back((Eigen::VectorXd(1) << 1).finished());
        targets.push_back((Eigen::VectorXd(1) << 0).finished());

        for(auto x : indata)
            cout << x << endl;
        cout << "targets\n";
        for(auto x : targets)
            cout << x << endl;
        cout << "\n\n\n";
        

        mm.train(indata, targets, 10, 100);

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

    init = new Normal(50, 20);
    inmat = Eigen::MatrixXd::Zero(2, 2);
    init->init_weights(inmat);
    delete(init);

    init = new Constant(44.0);
    inmat = Eigen::MatrixXd::Zero(2, 2);
    init->init_weights(inmat);
    delete(init);
    
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

    //confirmed as working
    act = new Tanh();
    indat = Eigen::VectorXd::Random(10);
    act->activate(indat);
    delete(act);

    return true;
}

int main(int argc, char **argv) {
    cout << "Starting Unit Tests :" << endl;
    cout << "Testing Initializers ... ";
    if(test_initializers()) cout << "Success" << endl;
    else cout << "Failed" << endl; 
    cout << "Testing Activations... ";
    if(test_activations()) cout << "Success" << endl;
    else cout << "Failed" << endl; 

    cout << "Testing Model Construction...";
    if( model_construct() && add_layers() )
        cout << "Success.\n";
    else 
        cout << "Failed\n";

    cout << "\n\n\n";

    simple_xor_test();
}
