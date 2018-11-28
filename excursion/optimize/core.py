import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm

def h_normal(s):
    return np.log(s * (2 * np.e * np.pi) ** 0.5)

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
        mi[~np.isfinite(mi)] = 0.0
        tocat.append(mi)

    return -np.mean(np.concatenate(tocat))
