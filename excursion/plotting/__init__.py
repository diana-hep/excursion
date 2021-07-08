import torch
import matplotlib.pyplot as plt

def plot_testcase_1d(testcase, tol):

    X = testcase.plot_X

    #plot functions
    for i, func in enumerate(testcase.true_functions):
        f = plt.plot(X, func(X), 'k--', label=f"true function \#{i}")

    #plot threshold s
    for thr in testcase.thresholds:
        min_X = torch.min(X)
        max_X = torch.max(X)
        plt.hlines(thr, min_X, max_X, colors = 'grey', label=f"threshold \#{i} at {thr.item():.2f}")

    #plot level sets


    ixs = torch.abs(func(X) - testcase.thresholds.item())< tol

    plt.plot(X[ixs], func(X[ixs]), 'r*', label='target level set')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('testcase 1D')
    plt.legend(loc=0)
    plt.show()


