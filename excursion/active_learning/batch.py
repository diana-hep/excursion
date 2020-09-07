import torch
import gpytorch
import numpy as np


def get_first_max_index(gp, ordered_indexs, testcase):
    X_train = gp.train_inputs[0]

    for i in ordered_indexs:
        if testcase.X.tolist()[i] not in X_train.tolist():
            new_first = i
            break

    return new_first


def get_naive_batch(gp, ordered_indexs, testcase, batchsize, self, **kwargs):
    X_train = gp.train_inputs[0]
    new_indexs = []
    new_ordered_indexs = ordered_indexs

    while len(new_indexs) < batchsize:
        for i in new_ordered_indexs:
            if testcase.X.tolist()[i] not in X_train.tolist() and i not in new_indexs:
                new_indexs.append(i)
                break

    return new_indexs


def get_kb_batch(gp, ordered_indexs, testcase, batchsize, self, **kwargs):
    X_train = gp.train_inputs[0]
    new_indexs = []
    new_ordered_indexs = ordered_indexs
    likelihood = kwargs['likelihood']
    algorithmopts = kwargs['algorithmopts']

    while(len(new_indexs) < batchsize):
        max_index = get_first_max_index(gp, new_ordered_indexs, testcase)
        new_indexs.append(max_index)

        x_new = testcase.X[max_index].reshape(1,-1)
        gp.eval()
        likelihood.eval()
        y_new = likelihood(gp(x_new)).mean

        self.x_new = x_new
        self.y_new = y_new

        gp = self.update_posterior(testcase, algorithmopts, gp, likelihood)
        new_ordered_indexs = self.get_ordered_indexs(gp, testcase)[0]
    
    return new_indexs


batch_types = {"Naive": get_naive_batch, "KB": get_kb_batch}

