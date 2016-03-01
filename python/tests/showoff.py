from mlstax import model, layer, loss

if __name__ == '__main__' :
    trainx, trainy, testx, testy = get_data("input.txt")
    mm = model.Model(input_dim=6)

    # add 4 layers, 3 dense and one recurrent
    mm.push_layer(layer.Dense(size=12, activation="relu"))
    mm.push_layer(layer.Dense(size=24, activation="sigmoid"))
    mm.push_layer(layer.RNN(size=24, mem_len=20)) 
    mm.push_layer(layer.Dense(size=10, activation="softmax"))

    opt = SGD(lr=0.1, momentum=0.025, nesterov=True)
    mm.compile(loss="rmse", optimizer=opt)

    mm.train(trainx, trainy, batch_size=100, nb_epochs=200)

    mm.evaluate(testx, testy, verbose=True)

    mm.save_weights("modelweights.dat")
    mm.save_model("modelarch.json") 
