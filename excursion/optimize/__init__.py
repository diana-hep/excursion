import numpy as np
import datetime
import torch
import gpytorch

from . import core


def gridsearch(gps, X, scandetails):
    thresholds = [-np.inf] + scandetails.thresholds + [np.inf]
    print('info_gain')
    acqval = np.array([core.info_gain(xtest, gps, thresholds, scandetails.meanX) for xtest in scandetails.acqX])

    newx = None
    for i,cacq in enumerate(scandetails.acqX[np.argsort(acqval)]):
        if cacq.tolist() not in X.tolist():
            print('taking new x. best non-existent index {} {}'.format(i,cacq))
            newx = cacq
            return newx,acqval
        else:
            print('{} is good but already there'.format(cacq))
    print('returning None.. something must be wrong')
    return None,None


def gridsearch_gpytorch(gps, X, scandetails):
    thresholds = [-np.inf] + scandetails.thresholds + [np.inf]
    print('info_gain')
    acqval = np.array([core.info_gain_gpytorch(xtest, gps, thresholds, scandetails.meanX) for xtest in scandetails.acqX])
    
    newx = None
    for i,cacq in enumerate(scandetails.acqX[np.argsort(acqval)]):
        if(cacq not in X.tolist()):
            print('taking new x. best non-existent index {} {}'.format(i,cacq))
            newx = cacq
            return newx,acqval
        else:
            print('{} is good but already there'.format(cacq))
    print('returning None.. something must be wrong')
    return None,None


def batched_gridsearch(gps, X, scandetails, gp_maker = None, batchsize=1):
    newX = np.empty((0,X.shape[-1]))
    my_gps    = gps
    my_y_list = [gp.y_train_ for gp in my_gps]
    myX       = X

    acqinfos = []

    while True:
        newx,acqinfo = gridsearch(my_gps, myX, scandetails)
        newX = np.concatenate([newX,np.asarray([newx])])
        acqinfos.append(acqinfo)
        if(len(newX)) == batchsize:
            print('we got our batch')
            return newX, acqinfos

        print('do fake update')
        myX = np.concatenate([myX,newX])

        newy_list = [gp.predict(newX) for gp in my_gps]
        for i,newy in enumerate(newy_list):
            print('new y i: {} {}'.format(i,newy))
            my_y_list[i] = np.concatenate([my_y_list[i],newy])
        print('build fake gps')
        my_gps = [gp_maker(myX,my_y) for my_y in my_y_list]


def train_hyperparameters(model, train_x, train_y, likelihood, optimizer):
    training_iter = 50
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    if(optimizer=='LBFGS'):
        optimizer = torch.optim.LBFGS([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1, line_search_fn='strong_wolfe')

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        def closure():
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            return loss

        for i in range(training_iter):
            optimizer.step(closure)


    if(optimizer=='Adam'):
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        