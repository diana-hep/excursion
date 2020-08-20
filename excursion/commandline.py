import sys
import os

sys.path.append(os.getcwd())

from excursion import init_gp
from excursion import ExcursionSetEstimator
import numpy as np
import importlib
import json
import yaml
import time
import torch
import datetime
import argparse


np.warnings.filterwarnings("ignore")


def load_example(example):
    testcase = None
    if example == "1Dtoyanalysis":
        testcase = importlib.import_module("testcases.fast_1D")
    elif example == "2Dtoyanalysis":
        testcase = importlib.import_module("testcases.fast_2D")
    elif example == "darkhiggs":
        testcase = importlib.import_module("excursion.testcases.darkhiggs")
    elif example == "checkmate":
        testcase = importlib.import_module("excursion.testcases.checkmate")
    elif example == "3dfoursheets":
        testcase = importlib.import_module("excursion.testcases.toy3d_foursheets")
    elif example == "3Dtoyanalysis":
        testcase = importlib.import_module("excursion.testcases.fast_3D")
    else:
        raise RuntimeError("unnkown test case")
    return testcase


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

    print("device", type(device))


    algorithmopts = yaml.safe_load(open(args.algorithm_specs, "r"))

    testcase = load_example(algorithmopts["example"])

    start_time = time.time() #######


    os.system("echo start_init_gp")
    model, likelihood = init_gp(testcase, algorithmopts, algorithmopts["ninit"], device)
    os.system("echo end_init_gp")
 
    time1=time.time()####
    print("--- init_gp %s seconds ---" % (time1 - start_time)) ###

    estimator = ExcursionSetEstimator(
        testcase, algorithmopts, model, likelihood, device
    )

    time2=time.time()####
    print("--- init_excursion %s seconds ---" % (time2 - time1)) ###

    timestampStr = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S") + "/"

    os.mkdir(args.outputfolder + timestampStr)

    while estimator.this_iteration < algorithmopts["nupdates"]:
        os.system("echo start_step")
        estimator.step(testcase, algorithmopts, model, likelihood)
        os.system("echo end_step")

        time3=time.time()####
        print("--- step %s seconds ---" % (time3 - time2)) ###


        model = estimator.update_posterior(testcase, algorithmopts, model, likelihood)
        
        time4=time.time()####
        print("--- posterior %s seconds ---" % (time4 - time3)) ###

        #estimator.plot_status(
        #    testcase, model, estimator.acq_values, args.outputfolder + timestampStr
        #)
        estimator.get_diagnostics(testcase, model, likelihood)

        time5=time.time()####
        print("--- get_diagnostics %s seconds ---" % (time5 - time4)) ###


    estimator.print_results(args.outputfolder + timestampStr, testcase, algorithmopts)


if __name__ == "__main__":
    main()
