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
    new_xs = torch.Tensor([])
    new_fake_ys = torch.Tensor([])

    new_ordered_indexs = ordered_indexs
    likelihood = kwargs["likelihood"]
    algorithmopts = kwargs["algorithmopts"]
    gp_fake = gp

    while len(new_indexs) < batchsize:
        max_index = get_first_max_index(gp, new_ordered_indexs, testcase)
        new_indexs.append(max_index)

        x = testcase.X[max_index].reshape(1, -1)
        new_xs = torch.cat((new_xs, x), 0)
        gp_fake.eval()
        likelihood.eval()
        y_fake = likelihood(gp_fake(x)).mean
        new_fake_ys = torch.cat((new_fake_ys,y_fake), 0)

        gp_fake = self.update_fake_posterior(testcase, algorithmopts, gp_fake, likelihood, new_xs, new_fake_ys)
        new_ordered_indexs = self.get_ordered_indexs(gp_fake, testcase)[0]

    return new_indexs


batch_types = {"Naive": get_naive_batch, "KB": get_kb_batch}
