#/usr/bin/env python
import numpy as np 

if __name__ == '__main__' :
    X = np.random.rand(10000, 5)*2
    Y = np.array([(x[0] + 0.35*x[1] + 0.5*x[2] + x[3] + 1*x[4]) for x in X])
    print "%s %s %s" % (5, 18, 1)
    for x in range(len(X)) : 
        print "%.2f %.2f %.2f %.2f %.2f %.2f" % (X[x][0], X[x][1], X[x][2], X[x][3], X[x][4], Y[x])
