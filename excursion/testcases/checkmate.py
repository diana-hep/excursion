import json
import numpy as np
import pkg_resources
import pickle

from .. import utils
from .. import ExcursionProblem

datafile = pkg_resources.resource_filename('excursion','testcases/data/checkmate_dense.json')

def modify(zv):
    return np.log(zv)-np.log(0.05)

truthX, truthy_obs, truthy_exp = [], [], []
for p,_,result in json.load(open(datafile))['precomputed']:
    if p[0] < p[1]+200: continue
    truthX.append(p)
    truthy_obs.append(max(float(result[1]['observed_CLs']),0.001) if result[1] else 0.001)
    truthy_exp.append(max(float(result[1]['expected_CLs']),0.001) if result[1] else 0.001)

truthX = np.array(truthX)

truthy_obs = np.array(truthy_obs)
truthy_obs = modify(truthy_obs)

truthy_exp = np.array(truthy_exp)
truthy_exp = modify(truthy_exp)


import sklearn.preprocessing
scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(truthX)
truthX = scaler.transform(truthX)

picklefile = pkg_resources.resource_filename('excursion','testcases/data/checkmate.pkl')
d = pickle.load(open(picklefile,'rb'))

def truth_obs(X):
    return 2*d['obs'].predict(X)

def truth_exp(X):
    return 2*d['exp'].predict(X)

thresholds = [modify(0.05)]
functions = [truth_obs, truth_exp]

def invalid_region(x):
    oX = scaler.inverse_transform(x)
    return oX[:,0] < oX[:,1] + 202

exp_and_obs = ExcursionProblem(
    functions, thresholds, ndim = 2,
    invalid_region = invalid_region,
    plot_npoints=[350,350]
)
