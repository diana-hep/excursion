import matplotlib
matplotlib.use('PS')

from . import diagnosis
from . import optimize
from . import get_gp
import numpy as np
import datetime
import click
import importlib
import json
import yaml

def diagnosis_set(gps,scandetails, X, y_list):
    confusion_list, predlabels_list, truelabels_list, diagX, labels = diagnosis.confusion_matrix(gps,scandetails)

    pct_correct = []
    for i in range(len(gps)):
        t = truelabels_list[i]
        c = confusion_list[i]
        pct_correct.append(np.sum([c[l-1][l-1]*len(t[t==l]) for l in labels ])/len(t))
    return {
        'pct_correct': pct_correct,
        'confusion_matrices': confusion_list
    }

def runloop(n_initialize, scandetails, n_updates, acq_optimizer = 'gridsearch', acqopts = None, gpopts = None):
    ndim = len(scandetails.plot_rangedef[:,0])

    initX = np.random.choice(range(len(scandetails.acqX)), size = n_initialize, replace=False)
    initX = scandetails.acqX[initX]
    nfuncs = len(scandetails.truth_functions)

    print("running loop for {} functions at thresholds {}".format(nfuncs, scandetails.thresholds))

    acqopts = acqopts or {}
    gpopts  = gpopts or {}

    loop_spec = {
        'initX': initX.tolist(),
        'n_updates': n_updates,
        'n_initialize': n_initialize,
        'acqopts': acqopts,
        'gpopts':gpopts
    }

    update_diagnoses = []

    X = initX
    y_list = [func(initX) for func in scandetails.truth_functions]

    gp_maker = lambda X,y: get_gp(X,y,**gpopts)

    gps = [gp_maker(X,y) for y in y_list]
    for index in range(n_updates):
        print('start acquisition {} at {}'.format(index,datetime.datetime.now()))

        if acq_optimizer=='gridsearch':
            newx, acqinfo = optimize.gridsearch(gps, X, scandetails,**acqopts)
            newX = np.asarray([newx])
        elif acq_optimizer=='batchedgrid':
            newX, acqinfo = optimize.batched_gridsearch(gps, X, scandetails, gp_maker=gp_maker, **acqopts)
        elif acq_optimizer=='gpsearch':
            newx, acqinfo = optimize.gpsearch(gps, X, scandetails,**acqopts)
            newX = np.asarray([newx])
        else:
            raise RuntimeError('unknown acquisition func optimizer {}'.format(acq_optimizer))

        print('end acquisition {} at {}'.format(index,datetime.datetime.now()))
        assert newX is not None

        print('new X {}'.format(newX))

        X = np.concatenate([X,newX])
        newy_list = [func(newX) for func in scandetails.truth_functions]
        for i,newy in enumerate(newy_list):
            print('new y i: {} {}'.format(i,newy))
            y_list[i] = np.concatenate([y_list[i],newy])

        update_diagnoses.append(diagnosis_set(gps, scandetails, X, y_list))

        all_results =  {
            'loop_spec': loop_spec,
            'update_diagnoses': update_diagnoses,
            'X': X.tolist(),
            'y': [y.tolist() for y in y_list]
        }
        print(all_results['update_diagnoses'][-1], len(X))
        gps = [gp_maker(X,y) for y in y_list]
        yield gps, acqinfo, all_results

def load_example(example):
    scandetails = None
    if example == '2Dtoyanalysis':
        scandetails = importlib.import_module('excursion.testcases.fast')
    elif example == 'darkhiggs':
        scandetails = importlib.import_module('excursion.testcases.darkhiggs')
    elif example == 'checkmate':
        scandetails = importlib.import_module('excursion.testcases.checkmate')
    elif example == '3dfoursheets':
        scandetails = importlib.import_module('excursion.testcases.toy3d_foursheets')
    elif example == '3Dtoyanalysis':
        scandetails = importlib.import_module('excursion.testcases.fast3d')
    else:
        raise RuntimeError('unnkown test case')
    return scandetails

@click.command()
@click.argument('example')
@click.argument('outputfile')
@click.option('--ninit', default = 10)
@click.option('--nupdates', default = 100)
@click.option('--gpopts', default = '')
@click.option('--acqtype', default = 'gridsearch')
@click.option('--acqopts', default = '')
def main(example,outputfile,ninit,nupdates,gpopts,acqtype, acqopts):
    scandetails = load_example(example)
    gpopts = yaml.load(gpopts)
    acqopts = yaml.load(acqopts)
    print(gpopts,acqopts)
    for i,(gps, acqinfo, r) in enumerate(runloop(ninit, scandetails, nupdates, acq_optimizer=acqtype, gpopts = gpopts, acqopts = acqopts)):
        print('dumping iteration {}'.format(i))
        json.dump(r,open(outputfile,'w'))



@click.command()
@click.argument('example')
@click.argument('sampler_type')
@click.argument('outputfile')
@click.option('--gpopts', default = '')
@click.option('--sampleropts', default = '')
def compare_samplers(example,sampler_type,outputfile,gpopts,sampleropts):
    from .samplers import regular_grid_generator, latin_hypercube_generator
    scandetails = load_example(example)
    gpopts = yaml.load(gpopts)
    gpopts = gpopts or {}
    sampleropts = yaml.load(sampleropts)

    sampler_dict = {
        'regular': regular_grid_generator,
        'latin': latin_hypercube_generator
    }
    sampler = sampler_dict[sampler_type]

    sampleropts = sampleropts or {}
    all_results = []
    print(sampleropts)
    for X,sample_info in sampler(scandetails, **sampleropts):
        y_list  = [func(X) for func in scandetails.truth_functions]
        try:
            gps = [get_gp(X,y,**gpopts) for y in y_list]

            results =  {
                'sample_info': sample_info,
                'diagnosis': diagnosis_set(gps, scandetails, X, y_list),
                'X': X.tolist(),
                'y': [y.tolist() for y in y_list]
            }
            print(results['diagnosis'], len(X), results['sample_info'])
            all_results.append(results)
        except ValueError:
            print(X,y_list)
        json.dump(all_results,open(outputfile,'w'))

if __name__ == '__main__':
    main()
