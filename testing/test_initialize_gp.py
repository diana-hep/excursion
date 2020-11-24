# test_initialize_gp.py

import torch
import yaml
from excursion import init_gp
from excursion.utils import load_example


def test_init_gp():

    device = torch.device("cpu")
    ninit = 1
    algorithmopts = yaml.safe_load(open("testing/algorithm_specs.yaml", "r"))

    # three toy examples
    for example in ["1Dtoyanalysis", "2Dtoyanalysis", "3Dtoyanalysis"]:
        print(example)
        testcase = load_example(example)
        gp, likelihood = init_gp(testcase, algorithmopts, ninit, device)

        assert type(gp.train_inputs[0]) != type(None)
        assert type(gp.train_targets) != type(None)
