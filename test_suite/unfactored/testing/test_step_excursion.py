# test_step_excursion.py

import torch
import yaml
from excursion import init_gp, ExcursionSetEstimator
from excursion.utils import load_example


def test_step_excursion():

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

        # one iteration only to test
        estimator.step(testcase, algorithmopts, gp, likelihood)

        assert type(estimator) != type(None)
        assert type(estimator.x_new) != type(None)
        assert type(estimator.y_new) != type(None)
        assert estimator.walltime_step != 0.0
        assert estimator.this_iteration == 1
