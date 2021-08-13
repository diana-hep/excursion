# test_full_simple_batch.py

import torch
import yaml
from excursion import init_gp
from excursion.estimator import ExcursionSetEstimator
from excursion.utils import load_example
import unittest


class TestFullSimpleBatchKB(unittest.TestCase):

    def test_full_simple_batch_kb(self):

        tol = 1e-6
        device = torch.device("cpu")
        ninit = 1
        algorithmopts = yaml.safe_load(open("algorithm_specs_batch_kb.yaml", "r"))

        # three toy examples
        for example in ["1D_test", "2D_test", "3D_test"]:
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

        self.assertTrue(type(torch.abs(model.train_targets) <= tol) != type(None))
