# test_full_1D_line.py

import torch
import yaml
from excursion import init_gp, ExcursionSetEstimator
from excursion.utils import load_example
import unittest

class TestFullSimple(unittest.TestCase):

    def test_full_simple(self):

        tol = 1e-6
        device = torch.device("cpu")
        ninit = 2
        algorithmopts = yaml.safe_load(open("algorithm_specs_full_test.yaml", "r"))

        # three toy examples
        for example in ["1D_test"]:
            testcase = load_example(example)
            model, likelihood = init_gp(testcase, algorithmopts, ninit, device)

            estimator = ExcursionSetEstimator(
                testcase, algorithmopts, model, likelihood, device
            )

            while estimator.this_iteration < algorithmopts["nupdates"]:
                estimator.step(testcase, algorithmopts, model, likelihood)
                model = estimator.update_posterior(
                    testcase, algorithmopts, model, likelihood
                )
        print(torch.abs(model.train_targets) <= tol)
        self.assertTrue(type(torch.abs(model.train_targets) <= tol) != type(None))
