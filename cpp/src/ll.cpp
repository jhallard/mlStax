
#include <mlstax/model>
#include <mlstax/dense>

#include <vector>

using namespace mlstax;

int main(int argc, char *argv[]) {
    
    Model mm = Model(12);
    auto init = std::make_shared<Normal>(0, 2.0);
    auto act = std::make_shared<Sigmoid>();
    auto layer = std::make_shared<Dense>(20, 12, init, act); 
    mm.push_layer(layer);
    auto init2 = std::make_shared<Uniform>(-2.0, 2.0);
    auto layer2 = std::make_shared<Dense>(24, 20, init2, act); 
    mm.push_layer(layer2);

    auto data_labels_train = parse_input_data("inputdata.txt");    
    std::vector<Eigen::VectorXd> traindat = data_labels_train.first;
    std::vector<Eigen::VectorXd> trainlabels = data_labels_train.second;

    auto data_labels_test = parse_input_data("testdata.txt");    
    std::vector<Eigen::VectorXd> testdat = data_labels_test.first;
    std::vector<Eigen::VectorXd> testlabels = data_labels_test.second;

    auto results =  mm.train(traindat, trainlabels, 50, 300);

    auto eval_results = mm.evaluate(testdat, testlabels);

    mm.save_model("../models/newmodel.json");

}
