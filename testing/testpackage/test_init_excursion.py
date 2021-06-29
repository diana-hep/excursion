# test_initialize_excursion.py

import torch
import yaml
from excursion import init_gp, ExcursionSetEstimator
from excursion.utils import load_example


def test_init_excursion():

    device = torch.device("cpu")
    ninit = 1
    algorithmopts = yaml.safe_load(open("testing/algorithm_specs.yaml", "r"))

    # three toy examples
    for example in ["1Dtoyanalysis", "2Dtoyanalysis", "3Dtoyanalysis"]:
        testcase = load_example(example)
        gp, likelihood = init_gp(testcase, algorithmopts, ninit, device)

        estimator = ExcursionSetEstimator(
            testcase, algorithmopts, gp, likelihood, device
        )

        assert type(estimator) != type(None)
        assert type(estimator._X_grid) != type(None)
        assert type(estimator._n_dims) != type(None)
        assert type(estimator._acq_type) != type(None)
