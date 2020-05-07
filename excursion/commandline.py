import matplotlib
matplotlib.use('PS')

from excursion import init_gp
from excursion.utils import get_first_max_index
from excursion import ExcursionSetEstimator
import excursion.metrics
import numpy as np
import click
import importlib
import json
import yaml
import time
import torch
import os
import datetime
import argparse

np.warnings.filterwarnings('ignore')

def load_example(example):
    testcase = None
    if(example == '1Dtoyanalysis'):
        testcase = importlib.import_module('testcases.fast_1D')
    elif(example == '2Dtoyanalysis'):
        testcase = importlib.import_module('testcases.fast_2D')
    elif(example == 'darkhiggs'):
        testcase = importlib.import_module('excursion.testcases.darkhiggs')
    elif(example == 'checkmate'):
        testcase = importlib.import_module('excursion.testcases.checkmate')
    elif(example == '3dfoursheets'):
        testcase = importlib.import_module('excursion.testcases.toy3d_foursheets')
    elif(example == '3Dtoyanalysis'):
        testcase = importlib.import_module('excursion.testcases.fast3d')
    else:
        raise RuntimeError('unnkown test case')
    return testcase



@click.command()
@click.argument('example')
@click.argument('outputfolder')
@click.option('--ninit', default = 10)
@click.option('--nupdates', default = 100)
@click.option('--algorithm_specs', default = '')
@click.option('--cuda', default = False)
def main(example,outputfolder,ninit,nupdates,algorithm_specs,cuda):
    if(not cuda and torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('device', device, type(device))

    testcase = load_example(example)

    algorithmopts = yaml.safe_load(open(algorithm_specs,'r'))

    model, likelihood = init_gp(testcase, algorithmopts, ninit, device)

    estimator = ExcursionSetEstimator(testcase, algorithmopts, model, likelihood, device)
    
    timestampStr = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")+'/'
    os.mkdir(outputfolder+timestampStr)

    while(estimator.this_iteration < nupdates):
        estimator.step(testcase, algorithmopts, model, likelihood)
        model = estimator.update_posterior(testcase, algorithmopts, model, likelihood, device)
        estimator.plot_status(testcase, model, estimator.acq_values, outputfolder+timestampStr)
        estimator.get_diagnostics(testcase, model,likelihood)

    estimator.print_results(outputfolder+timestampStr, testcase, algorithmopts)

if __name__ == '__main__':
    main()
