import numpy as np
from .. import ExcursionProblem

def func(X):
    f = 30
    return 11-10*(np.tanh((X-0.3)*3) + 0.15*np.sin(X*f))

single_function = ExcursionProblem([func],thresholds=[0.7])
