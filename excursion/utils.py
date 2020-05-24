from scipy.stats import norm
import numpy as np
import torch

def point_entropy(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf],thresholds,[np.inf]])

    entropies = []
    for mu,std in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(mu, std).cdf(thresholds[j+1]) - norm(mu, std).cdf(thresholds[j])
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1-1e-9] = 1-1e-9
            entropy -= p_within * np.log(p_within)  
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)


def point_entropy_gpytorch(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf],thresholds,[np.inf]])

    entropies = []
    for obs_pred in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()).cdf(thresholds[j+1]) - \
                       norm(obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()).cdf(thresholds[j])
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1-1e-9] = 1-1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)



def mesh2points(grid,npoints_tuple):
    ndim = len(npoints_tuple)
    X = np.moveaxis(grid,0,ndim).reshape(int(np.product(npoints_tuple)),ndim)
    return X

def points2mesh(X,npoints_tuple):
    ndim = len(npoints_tuple)
    grid = np.moveaxis(X.reshape(*(npoints_tuple +[ndim,])),ndim,0)
    return grid

def mgrid(rangedef):
    _rangedef = np.array(rangedef, dtype='complex128')
    slices = [slice(*_r) for _r in _rangedef]
    return np.mgrid[slices]


def values2mesh(values, rangedef, invalid, invalid_value = np.nan):
    grid = mgrid(rangedef)
    allX = mesh2points(grid,rangedef[:,2])
    allv = np.zeros(len(allX))
    inv  = invalid(allX)

    if(torch.cuda.is_available() and type(values)==torch.Tensor):
        allv[~inv]  = values.cpu()
    else:
        allv[~inv]  = values

    if np.any(inv):
        allv[inv]  = invalid_value
    return allv.reshape(*map(int,rangedef[:,2]))


def get_first_max_index(gp, new_index, testcase):
    X_train = gp.train_inputs[0]

    for i in new_index:
        if(testcase.X.tolist()[i] not in X_train.tolist()):
            new_first = i
            break
            
    return new_first
   

def h_normal(var):
    return torch.log(var * (2 * np.e * np.pi) ** 0.5)

def normal_pdf(x):
    return 1./(2*np.pi)**0.5 * torch.exp(-0.2*x**2)


def truncated_std_conditional(Y_pred_all, a, b):
    mu_grid = Y_pred_all.mean[1:]
    std_grid = Y_pred_all.variance[1:]**0.5
    mu_candidate = Y_pred_all.mean[0]
    std_candidate = Y_pred_all.variance[0]**0.5
    rho = Y_pred_all.covariance_matrix[0,1:] / (std_candidate*std_grid)

    #norm needs to be a normal distribution but in python
    normal = torch.distributions.Normal(loc=0, scale=1)
    alpha = (a - mu_grid) / std_grid
    beta = (b - mu_grid) / std_grid
    c = normal.cdf(beta) - normal.cdf(alpha)
    
    # phi(beta) = normal(0,1) at x = beta
    beta_phi_beta = beta*normal_pdf(beta)
    beta_phi_beta[~torch.isfinite(beta_phi_beta)] = 0.
    alpha_phi_alpha = alpha*normal_pdf(alpha)
    alpha_phi_alpha[~torch.isfinite(alpha_phi_alpha)] = 0.

    #unnormalized 
    first_moment = mu_candidate - std_candidate * rho / c * (normal_pdf(beta) - normal_pdf(alpha)) 

    second_moment = std_candidate**2 * (1-rho**2/c)*(beta_phi_beta - alpha_phi_alpha) \
                    - mu_candidate**2 +2*mu_candidate*first_moment

    return second_moment**0.5

