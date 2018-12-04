import excursion
import excursion.testcases.fast as scandetails
import excursion.optimize
import numpy as np
import logging

def test_2d():
    scandetails.truth_functions = [
        scandetails.truth,
    ]

    N_INIT    = 5
    N_UPDATES = 1
    N_BATCH   = 5
    N_DIM     = 2

    X,y_list, gps = excursion.optimize.init(scandetails, n_init = N_INIT, seed = 1)

    index = 0
    for index in range(1,N_UPDATES+1):
        newX, acqvals = excursion.optimize.gridsearch(gps, X, scandetails, batchsize=N_BATCH)
        newys_list = [func(np.asarray(newX)) for func in scandetails.truth_functions]
        for i,newys in enumerate(newys_list):
            y_list[i] = np.concatenate([y_list[i],newys])
        X = np.concatenate([X,newX])
        gps = [excursion.get_gp(X,y_list[i]) for i in range(len(scandetails.truth_functions))]
    print(X,X.shape)
    assert X.shape == (N_INIT + N_BATCH * N_UPDATES,N_DIM)
    assert np.allclose(X[0],[6.25533007e-01, 1.08048674e+00])
