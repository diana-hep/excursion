import numpy as np

def truthfunc(X):
    f = 15
    return 10-10*(np.tanh(X*2) + 0.15*np.sin(X*f))

truth_functions = [truthfunc]
thresholds = [0.7]

plot_rangedef = np.asarray([
    [-1,1,100]
])
plotX = np.linspace(*plot_rangedef[0]).reshape(-1,1)

acq_rangedef = np.asarray([
    [-1,1,100]
])
acqX = np.linspace(*acq_rangedef[0]).reshape(-1,1)



mean_rangedef = np.asarray([
    [-1,1,100]
])
meanX = np.linspace(*mean_rangedef[0]).reshape(-1,1)
ndims = 1
y_lim = [-5,30]
    
    