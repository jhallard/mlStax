#/usr/bin/env python
import numpy as np 
import sys

if __name__ == '__main__' :
    numdim = 5
    if len(sys.argv) > 1 :
        numdim = int(sys.argv[1])
    X = np.random.rand(16*10000, numdim)
    F = np.random.rand(1, numdim)*2
    Y = np.dot(X, F.T)
    print "%s %s %s %s %s" % (numdim, 24, 48, 32,  1)
    for x in range(len(X)) : 
        s = " ".join(["%.5f" % X[x][i] for i in range(len(X[x]))])
        s += " %.5f" % Y[x]
        print s
        # print "%.2f %.2f %.2f %.2f %.2f %.2f" % (X[x][0], X[x][1], X[x][2], X[x][3], X[x][4], Y[x])
