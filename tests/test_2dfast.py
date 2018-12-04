import excursion
import excursion.testcases.fast as scandetails
import excursion.optimize
import numpy as np
import logging

def test_2d():
    scandetails.truth_functions = [
        scandetails.truth,
    ]

    N_DIM     = 2
    N_INIT    = 5
    N_FUNCS   = 1
    N_UPDATES = 1
    N_BATCH   = 5

    np.random.seed(1)
    X = np.random.uniform(scandetails.plot_rangedef[:,0],scandetails.plot_rangedef[:,1], size = (N_INIT,N_DIM))
    y_list = [np.array([scandetails.truth_functions[i](np.asarray([x]))[0] for x in X]) for i in range(N_FUNCS)]

    gps = [excursion.get_gp(X,y_list[i]) for i in range(N_FUNCS)]


    index = 0
    for index in range(1,N_UPDATES+1):
        newX, acqvals = excursion.optimize.gridsearch(gps, X, scandetails, batchsize=N_BATCH)
        newys_list = [scandetails.truth_functions[i](np.asarray(newX)) for i in range(N_FUNCS)]
        for i,newys in enumerate(newys_list):
            y_list[i] = np.concatenate([y_list[i],newys])
        X = np.concatenate([X,newX])
        gps = [excursion.get_gp(X,y_list[i]) for i in range(N_FUNCS)]
    print(X,X.shape)
    assert X.shape == (N_INIT + N_BATCH * N_UPDATES,N_DIM)
