import sys
import os

sys.path.append(os.getcwd())

from excursion import init_gp
from excursion import ExcursionSetEstimator
from excursion.utils import load_example
import numpy as np
import yaml
import time
import torch
import datetime
import argparse

np.warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Description of excursion job")

parser.add_argument(
    "--outputfolder", required=True, metavar="OUT", type=str, help="outputfolder path"
)

parser.add_argument(
    "--algorithm_specs",
    required=True,
    metavar="ALG",
    type=str,
    default="",
    help="path to yaml file with algorithm specifications",
)

parser.add_argument(
    "--cuda", required=False, metavar="CUDA", type=bool, default=False, help="use GPU"
)

args = parser.parse_args()


def main():
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device", type(device))

    algorithmopts = yaml.safe_load(open(args.algorithm_specs, "r"))

    testcase = load_example(algorithmopts["example"])

    # start_time = time.time()  #######

    models, likelihood = init_gp(testcase, algorithmopts, algorithmopts["ninit"], device)

    # time1 = time.time()  ####
    # print("--- init_gp %s seconds ---" % (time1 - start_time))  ###

    estimator = ExcursionSetEstimator(
        testcase, algorithmopts, models, likelihood, device
    )

    timestampStr = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S") + "/"

    os.makedirs(args.outputfolder + timestampStr)

    while estimator.this_iteration < algorithmopts["nupdates"]:
        time2 = time.time()

        estimator.step(testcase, algorithmopts, models, likelihood)

        time3 = time.time()  ####
        print("--- step %s seconds ---" % (time3 - time2))  ###

        models = estimator.update_posterior(testcase, algorithmopts, models, likelihood)

        time4 = time.time()  ####
        print("--- posterior %s seconds ---" % (time4 - time3))  ###

        estimator.plot_status(
            testcase,
            algorithmopts,
            models,
            estimator.acq_values,
            args.outputfolder + timestampStr,
        )
        estimator.get_diagnostics(testcase, models, likelihood)

        time5 = time.time()  ####
        print("--- get_diagnostics %s seconds ---" % (time5 - time4))  ###

    estimator.print_results(args.outputfolder + timestampStr, testcase, algorithmopts)


if __name__ == "__main__":
    main()
