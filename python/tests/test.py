#!/usr/bin/env python

"""
Simple testing file for the DNNs. Build a few models, give them some data,
make sure things don't crash and burn.
"""

import numpy as np
import sys
from sys import stdin

sys.path.append("../")
sys.path.append("../mlstax")
sys.path.append("../../")

import model, layer, activations, initializers

def num(x) :
    try:
        return int(x)
    except ValueError:
        return float(x)

if __name__ == '__main__' :
    fn = "input.txt"
    indat = []
    targets = []
    indim = outdim = 0
    layer_dims = []
    epochs = 5
    if len(sys.argv) > 1 :
        fn = sys.argv[1]
    if len(sys.argv) > 2 :
        epochs = int(sys.argv[2])

    # collect data from input file
    # format should be 
    # INSIZE L1_SIZE L2_SIZE .... LN_SIZE OUTSIZE
    # feature1_1 feature1_2 feature1_3 ... feature1_m target1_1 ... target1_k
    # feature2_1 feature2_2 feature2_3 ... feature2_m target2_1 ... target2_k
    # ..... (list all data as features targets)
    with open(fn) as fh :
        for x, line in enumerate(fh.readlines()) :
            if x == 0 : 
                vals = line.split(' ')
                vals = [num(x) for x in vals]
                if len(vals) < 3 :
                    print "Need at least 3 layers, input, hidden, output"
                    sys.exit(1)
                indim = vals[0]
                outdim = vals[-1]
                layer_dims = vals[1:-1]
                continue
            else :
                vals = line.split(' ')
                vals = [num(x) for x in vals]
                indat.append(vals[:indim])
                targets.append(vals[indim:])

    indat = np.array(indat)
    targets = np.array(targets)

    # print "Input Data : %s \nOutput Data : %s\n" % (indat, targets)

    # build the model
    mm = model.Model(indim)
    for ldim in layer_dims :
        mm.push_layer(layer.Dense(ldim))
    mm.push_layer(layer.Dense(outdim, activation=activations.Nothing))

    print str(mm)

    print "Compiling Model\n\n"
    mm.compile()

    print "Beginning training session\n\n"

    mm.train(indat, targets, 1, epochs)

    print "Submit inputs for evaluation (1 per line, then EOF)\n\n"
    for line in sys.stdin:
        inp = [float(x) for x in line.strip().split(' ')]
        print mm.predict(np.array([inp]))

