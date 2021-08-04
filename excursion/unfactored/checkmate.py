import json
import numpy as np
import pkg_resources
import pickle
from excursion.utils import mgrid, mesh2points
import torch
import sklearn.preprocessing


datafile = pkg_resources.resource_filename(
    "excursion", "unfactored/data/checkmate_dense.json"
)

def modify(zv):
    return np.log(zv) - np.log(0.05)


truthX, truthy_obs, truthy_exp = [], [], []
for p, _, result in json.load(open(datafile))["precomputed"]:
    if p[0] < p[1] + 200:
        continue
    truthX.append(p)
    truthy_obs.append(
        max(float(result[1]["observed_CLs"]), 0.001) if result[1] else 0.001
    )
    truthy_exp.append(
        max(float(result[1]["expected_CLs"]), 0.001) if result[1] else 0.001
    )

truthX = np.array(truthX)
truthy_obs = np.array(truthy_obs)
truthy_obs = modify(truthy_obs)

truthy_exp = np.array(truthy_exp)
truthy_exp = modify(truthy_exp)


scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(truthX)
truthX = scaler.transform(truthX)

picklefile = pkg_resources.resource_filename(
    "excursion", "unfactored/data/checkmate.pkl"
)
d = pickle.load(open(picklefile, "rb"))

print('d', type(d))
print(d)


def truth_obs(X):
    from scipy.interpolate import griddata
    grid_data = torch.from_numpy(griddata(truthX,truthy_obs,X))
    return grid_data




def invalid_region(x):
    oX = scaler.inverse_transform(x)
    return oX[:, 0] < oX[:, 1] + 202


thresholds = torch.Tensor([0.05])
true_functions = [truth_obs]
n_dims = 2

rangedef = np.array([[0.0, 1.0, 101], [0.0, 1.0, 101]])
plot_meshgrid = mgrid(rangedef)
X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
X = torch.from_numpy(X_plot)
