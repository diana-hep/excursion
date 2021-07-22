from scipy.stats import norm
import numpy as np

def point_entropy(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    entropies = []
    for mu, std in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(mu, std).cdf(thresholds[j + 1]) - norm(mu, std).cdf(
                thresholds[j]
            )
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1 - 1e-9] = 1 - 1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)


def point_entropy_gpytorch(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    entropies = []
    for obs_pred in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(
                obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()
            ).cdf(thresholds[j + 1]) - norm(
                obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()
            ).cdf(
                thresholds[j]
            )
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1 - 1e-9] = 1 - 1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)

