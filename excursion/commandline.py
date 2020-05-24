import matplotlib
matplotlib.use('PS')

import sys
import os
sys.path.append(os.getcwd()) 


from excursion import init_gp
from excursion.utils import get_first_max_index
from excursion import ExcursionSetEstimator
import excursion.metrics
import numpy as np
import importlib
import json
import yaml
import time
import torch
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


parser = argparse.ArgumentParser(description='Description of excursion job')

parser.add_argument('--example', required=True, metavar='EX', type=str, 
                    help='example to load')

parser.add_argument('--outputfolder', required=True,  metavar='OUT', type=str, 
                    help='outputfolder path')

parser.add_argument('--ninit', required=False, metavar='NINIT', type=int, default=10,  
                    help='number of init training points')

parser.add_argument('--nupdates', required=False, metavar='NUP', type=int, default=100,  
                    help='number of iterations')

parser.add_argument('--algorithm_specs', required=False, metavar='ALG', type=str, default='', 
                    help='path to yaml file with algorithm specifications')

parser.add_argument('--cuda', required=False, metavar='CUDA', type=bool, default=False, 
                    help='use GPU')

args = parser.parse_args()

def main():
    if(args.cuda and torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('device', device, type(device))

    testcase = load_example(args.example)

    algorithmopts = yaml.safe_load(open(args.algorithm_specs,'r'))

    model, likelihood = init_gp(testcase, algorithmopts, args.ninit, device)

    estimator = ExcursionSetEstimator(testcase, algorithmopts, model, likelihood, device)
    
    timestampStr = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")+'/'
    
    os.mkdir(args.outputfolder+timestampStr)

    while(estimator.this_iteration < args.nupdates):
        estimator.step(testcase, algorithmopts, model, likelihood)
        model = estimator.update_posterior(testcase, algorithmopts, model, likelihood)
        estimator.plot_status(testcase, model, estimator.acq_values, args.outputfolder+timestampStr)
        estimator.get_diagnostics(testcase, model,likelihood)

    estimator.print_results(args.outputfolder+timestampStr, testcase, algorithmopts)

if __name__ == '__main__':
    main()
