import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm
import gpytorch
import torch

def h_normal(s):
    return np.log(s * (2 * np.e * np.pi) ** 0.5)

def h_normal_gpytorch(s):
    return torch.log(s * (2 * np.e * np.pi) ** 0.5)

def approx_mi_vec(mu, cov, thresholds):
    mu1 = mu[:, 0]
    std1 = cov[:, 0, 0] ** 0.5
    mu2 = mu[:, 1]
    std2 = cov[:, 1, 1] ** 0.5
    rho = cov[:, 0, 1] / (std1 * std2)

    std_sx = []

    for j in range(len(thresholds) - 1):
        alpha_j = (thresholds[j] - mu2) / std2
        beta_j = (thresholds[j+1] - mu2) / std2
        c_j = norm.cdf(beta_j) - norm.cdf(alpha_j)

        # \sigma(Y(X)|S(x')=j)
        b_phi_b = beta_j * norm.pdf(beta_j)
        b_phi_b[~np.isfinite(beta_j)] = 0.0
        a_phi_a = alpha_j * norm.pdf(alpha_j)
        a_phi_a[~np.isfinite(alpha_j)] = 0.0

        mu_cond = mu1 - std1 * rho / c_j * (norm.pdf(beta_j) - norm.pdf(alpha_j))
        var_cond = (mu1 ** 2 - 2 * mu1 * std1 * (rho / c_j * (norm.pdf(beta_j) - norm.pdf(alpha_j))) +
                    std1 ** 2 * (1. - (rho ** 2 / c_j) * (b_phi_b - a_phi_a)) -
                    mu_cond ** 2)
        std_sx_j = var_cond ** 0.5

        std_sx.append(std_sx_j)

    # Entropy
    h = h_normal(std1)

    for j in range(len(thresholds) - 1):
        p_j = norm(mu2, std2).cdf(thresholds[j+1]) - norm(mu2, std2).cdf(thresholds[j])
        dec = p_j * h_normal(std_sx[j])
        h[p_j > 0.0] -= dec[p_j > 0.0]

    return h


def approx_mi_vec_gpytorch(mu, cov, thresholds):
    mu1 = mu[:, 0]
    std1 = cov[:, 0, 0] ** 0.5
    mu2 = mu[:, 1]
    std2 = cov[:, 1, 1] ** 0.5
    rho = cov[:, 0, 1] / (std1 * std2)

    std_sx = []

    for j in range(len(thresholds) - 1):
        alpha_j = (thresholds[j] - mu2) / std2
        beta_j = (thresholds[j+1] - mu2) / std2
        alpha_j = alpha_j.detach().numpy()
        beta_j = beta_j.detach().numpy()

        c_j = norm.cdf(beta_j) - norm.cdf(alpha_j)

        # \sigma(Y(X)|S(x')=j)
        b_phi_b = beta_j * norm.pdf(beta_j)
        b_phi_b[~np.isfinite(beta_j)] = 0.0
        a_phi_a = alpha_j * norm.pdf(alpha_j)
        a_phi_a[~np.isfinite(alpha_j)] = 0.0

        alpha_j = torch.tensor(alpha_j)
        beta_j = torch.tensor(beta_j)

        mu_cond = mu1 - std1 * rho / torch.tensor(c_j) * ( torch.tensor(norm.pdf(beta_j)) - torch.tensor(norm.pdf(alpha_j)) )
        var_cond = (mu1 ** 2 - 2 * mu1 * std1 * (rho / torch.tensor(c_j) * ( torch.tensor(norm.pdf(beta_j)) - torch.tensor(norm.pdf(alpha_j))) ) +
                    std1 ** 2 * (1. - (rho ** 2 / torch.tensor(c_j)) * (torch.tensor(b_phi_b) - torch.tensor(a_phi_a))) -
                   mu_cond ** 2)

        std_sx_j = var_cond ** 0.5

        std_sx.append(std_sx_j)

    # Entropy
    h = h_normal_gpytorch(std1)

    for j in range(len(thresholds) - 1):
        p_j = norm(mu2.detach().numpy(), std2.detach().numpy()).cdf(thresholds[j+1]) - \
              norm(mu2.detach().numpy(), std2.detach().numpy()).cdf(thresholds[j])
        dec = torch.tensor(p_j) * h_normal_gpytorch(std_sx[j])
        h[p_j > 0.0] -= dec[p_j > 0.0]

    return h


def info_gain(x_candidate, gps, thresholds, meanX):
    # Slow
    # mus, covs = [], []
    # for x in meanX:
    #     mu, cov = gp.predict(np.array([x_candidate, x]), return_cov=True)
    #     mus.append(mu)
    #     covs.append(cov)
    #
    # mus = np.array(mus)
    # covs = np.array(covs)

    # Fast
    n_samples = len(meanX)
    X_all = np.concatenate([np.array([x_candidate]), meanX]).reshape(1 + n_samples, -1)
    tocat = []
    for gp in gps:
        K_trans_all = gp.kernel_(X_all, gp.X_train_)

        ########
        print('K_trans_all = gp.kernel_(X_all, gp.X_train_) \n')
        print('shape X_all ',  X_all.shape)
        #print( X_all)
        print('shape gp.X_train_ ',  gp.X_train_.shape)
        print( gp.X_train_)

        print('\n')
        print('X_all multiplied gp.X_train_')
        dot = np.matmul(X_all,gp.X_train_.T)
        print( dot )
        print('\n')

        print('\n')
        print('RBF (X_all multiplied gp.X_train_)')
        print( dot )
        print('\n')

        print('shape K_trans_all ', K_trans_all.shape)
        print(K_trans_all)
        ####


        y_mean_all = K_trans_all.dot(gp.alpha_) + gp._y_train_mean
        v_all = cho_solve((gp.L_, True), K_trans_all.T)

        mus = np.zeros((n_samples, 2))
        mus[:, 0] = y_mean_all[0]
        mus[:, 1] = y_mean_all[1:]

        covs = np.zeros((n_samples, 2, 2))
        c = gp.kernel_(X_all[:1], X_all)
        covs[:, 0, 0] = c[0, 0]
        covs[:, 1, 1] = c[0, 0]
        covs[:, 0, 1] = c[0, 1:]
        covs[:, 1, 0] = c[0, 1:]

        K_trans_all_repack = np.zeros((n_samples, 2, len(gp.X_train_)))
        K_trans_all_repack[:, 0, :] = K_trans_all[0, :]
        K_trans_all_repack[:, 1, :] = K_trans_all[1:]
        v_all_repack = np.zeros((n_samples, len(gp.X_train_), 2))
        v_all_repack[:, :, 0] = v_all[:, 0]
        v_all_repack[:, :, 1] = v_all[:, 1:].T
        covs -= np.einsum('...ij,...jk->...ik', K_trans_all_repack, v_all_repack)

        mi = approx_mi_vec(mus, covs, thresholds)

        print('shape mi ', mi.shape)
        print(mi)
        mi[~np.isfinite(mi)] = 0.0 #just to avoid NaN
        print('mi after mi[~np.isfinite(mi)] = 0.0 ')
        print(mi)
        tocat.append(mi)

    return -np.mean(np.concatenate(tocat))


def info_gain_gpytorch(x_candidate, gps, thresholds, meanX):

    # Fast
    n_samples = len(meanX)
    X_all = torch.tensor(np.concatenate([np.array([x_candidate]), meanX]).reshape(1 + n_samples, -1))
    tocat = []

    for gp in gps:
        kernel = gp.covar_module
        # (train_inputs/targets,) tuple dont know behaviour with more train data iterations
        X_train = torch.tensor(gp.train_inputs[0])
        y_train = torch.tensor(gp.train_targets)
        y_train = y_train.view(len(y_train),1) #add dimension

        K_trans_all = kernel( X_all, X_train ) #lazy tensor, for matrix use .evaluate() 

        K_train = kernel( X_train, X_train)
        L = K_train.cholesky(upper=False)
        alpha = torch.cholesky_solve( y_train, L.evaluate(), upper=False )  #torch.cholesky_solve(v, L)

        y_mean_all = torch.matmul(K_trans_all.evaluate(), alpha) + torch.mean(y_train)
        v_all = torch.cholesky_solve(K_trans_all.t().evaluate(), L.evaluate())

        mus = torch.zeros((n_samples, 2)) 
        mus[:, 0] = y_mean_all[0]
        mus[:, 1] = y_mean_all[1:].squeeze()

        covs = torch.zeros((n_samples, 2, 2))
        c = kernel(X_all[:1], X_all)
        covs[:, 0, 0] = c[0, 0]
        covs[:, 1, 1] = c[0, 0]
        covs[:, 0, 1] = c[0, 1:]
        covs[:, 1, 0] = c[0, 1:]

        K_trans_all_repack = torch.zeros((n_samples, 2, len(X_train)))
        K_trans_all_repack[:, 0, :] = K_trans_all[0, :]

        K_trans_all_repack[:, 1, :] = K_trans_all.evaluate()[1:]
        v_all_repack = torch.zeros((n_samples, len(X_train), 2))
        v_all_repack[:, :, 0] = v_all[:, 0]
        v_all_repack[:, :, 1] = v_all[:, 1:].T
        covs -= torch.einsum('...ij,...jk->...ik', K_trans_all_repack, v_all_repack)

        mi = approx_mi_vec_gpytorch(mus, covs, thresholds)
        mi[~torch.isfinite(mi)] = 0.0
        tocat.append(mi)

    return -torch.mean(torch.cat(tocat))
