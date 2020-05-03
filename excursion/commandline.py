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
#@click.option('--cuda', default = False)
def main(example,outputfolder,ninit,nupdates,algorithm_specs):
    #if(not cuda and torch.cuda.is_available()):
    #    args.device = torch.device('cuda')
    #else:
    #    args.device = torch.device('cpu')

    testcase = load_example(example)

    algorithmopts = yaml.safe_load(open(algorithm_specs,'r'))

    model, likelihood = init_gp(testcase, algorithmopts, ninit)

    estimator = ExcursionSetEstimator(testcase, algorithmopts, model, likelihood)
    
    timestampStr = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")+'/'
    os.mkdir(outputfolder+timestampStr)

    while(estimator.this_iteration < nupdates):
        estimator.step(testcase, algorithmopts, model, likelihood)
        model = estimator.update_posterior(testcase, algorithmopts, model, likelihood)
        estimator.plot_status(testcase, model, estimator.acq_values, outputfolder+timestampStr)
        estimator.get_diagnostics(testcase, model,likelihood)

    estimator.print_results(outputfolder+timestampStr, testcase, algorithmopts)
if __name__ == '__main__':
    main()



























#################################################################

#def runloop(n_initialize, testcase, n_updates, algorithmopts):
#     ndim = testcase.n_dims

#     # init training data
#     init_type = algorithmopts['init_type']
#     initX = init_traindata(testcase, init_type, n_initialize) 

#     X_grid = testcase.X_plot
#     print('X_grid ', X_grid.shape)

#     nfuncs = len(testcase.true_functions)

#     print("running loop for {} functions at thresholds {}".format(nfuncs, testcase.thresholds))

#     loop_spec = {
#         'initX': initX,
#         'n_updates': n_updates,
#         'n_initialize': n_initialize,
#         'algorithmopts': algorithmopts
#     }

#     X = initX
#     y_list = [func(initX) for func in testcase.true_functions]
#     print('y_list ', len(y_list))
#     print(y_list)

#     gps = [get_gp(X,y,**algorithmopts) for y in y_list]

#     #iteration samples 
#     acq_type = algorithmopts['acq']['acq_type']
#     update_diagnoses = []

#     for i in range(1,n_updates+1):

#         print('start acquisition {} at {}'.format(i, datetime.datetime.now()))

#         all_targets = []
#         all_inputs = []

#         #for every type of gp/model
#         for model in gps:
#             acquisition_values_grid = []
#             for x in X_grid:
#                 x_candidate = torch.Tensor([x]).double()
#                 print('x ',x_candidate.shape)
#                 value = acq(model,testcase, x_candidate, acq_type)
#                 acquisition_values_grid.append(value)
            
#             new_indexs = np.argsort(acquisition_values_grid)[::-1] #descending order
            
#             ##discard those points already in dataset
#             new_index = get_first_max_index(model, new_indexs, testcase)
        
#             ##get x, y
#             x_new = testcase_details.X[new_index].reshape(1,-1)
#             y_new = testcase_details.true_functions[0](x_new)
        
#             #update dataset
#             inputs_i = torch.cat((model.train_inputs[0], x_new),0)
#             targets_i = torch.cat((model.train_targets.view(-1,), y_new),0).flatten()

#             model.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)
#             model = get_gp(inputs_i, targets_i, **algorithmopts)
#             all_inputs.append(inputs_i)
#             all_targets.append(targets_i)
        
#         rundiagnosis(gps, testcase, all_inputs, all_targets)

#         #update_diagnoses.append(rundiagnosis(gps, testcase, all_inputs, all_targets))
#         ###NEXT
#         #all_results =  {
#         #    'loop_spec': loop_spec,
#         #    'update_diagnoses': update_diagnoses,
#         #    'X': X.tolist(),
#         #    'y': [y.tolist() for y in y_list]
#         #}
#         #print(all_results['update_diagnoses'][-1], len(X))
#         #gps = [get_gp(X,y,**algorithmopts) for y in y_list]
#         #yield gps, acquisition_values_grid, all_results
