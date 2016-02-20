
#include <vector>
#include <map>

class Model {

public :
    Model(int input_dim, std::vector<Layer> layers = {});

    bool push_layer(Layer layer);


private :

    int input_dim;


}
