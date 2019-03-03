import numpy as np
import datetime
import click
import importlib
import json
import time
import yaml
import logging
from . import Learner

import os

np.warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

LOGFORMAT = '%(asctime)s | %(name)20.20s | %(levelname)6s | %(message)s'
formatter = logging.Formatter(LOGFORMAT)

log = logging.getLogger(__name__)

EXAMPLES = ['2Dtoyanalysis', '3Dtoyanalysis', 'stopsearch', 'darkhiggs']
def load_example(example):
    if example == '2Dtoyanalysis':
        m = importlib.import_module('excursion.testcases.fast')
        return m.two_functions
    elif example == '3Dtoyanalysis':
        m = importlib.import_module('excursion.testcases.fast3d')
        return m.single_function
    elif example == 'stopsearch':
        m = importlib.import_module('excursion.testcases.checkmate')
        return m.exp_and_obs
    elif example == 'darkhiggs':
        m = importlib.import_module('excursion.testcases.darkhiggs')
        return m.iso_xsec
    else:
        raise RuntimeError('unnkown test case')
    return scandetails

def run_loop(learner,n_batch, n_updates):
    for index in range(n_updates):
        newX =  learner.suggest(batchsize=n_batch)
        learner.evaluate_and_tell(newX)
        yield learner

def load_learner(inputfile = None, init_config = None):
    if not inputfile:
        example, n_init = init_config
        log.debug('initializing')
        assert n_init
        learner = Learner(load_example(example))
        learner.initialize(n_init = n_init)
        return learner

def setup_logging(logfile):
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(level = logging.DEBUG, format = LOGFORMAT)
    root = logging.getLogger()
    root.handlers[0].setLevel(logging.INFO)
    fh  = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

@click.command()
@click.argument('example', type = click.Choice(EXAMPLES))
@click.argument('outputfile')
@click.option('--ninit', default = 10)
@click.option('--nbatch', default = 1)
@click.option('--nupdates', default = 100)
@click.option('--logfile', default = 'excursion.log')
def main(example, outputfile, logfile, ninit, nbatch, nupdates):
    setup_logging(logfile)

    learner = load_learner(init_config = (example, ninit))
    start = time.time()
    for idx,l in enumerate(run_loop(learner, nbatch, nupdates)):
        delta = time.time()-start
        start = time.time()
        msg = 'iteration {:5.0f} | misclass {:.3E} | npoints {:5.0f} | time {:10.2f} s | {}'.format(
            len(learner.metrics)-1,
            1-learner.metrics[-1]['confusion']['t'],
            learner.metrics[-1]['npoints'],
            delta,
            outputfile
        )
        log.info(msg)
    with open(outputfile,'w') as outfile:
        json.dump({'metrics': learner.metrics, 'X': learner.X.tolist(), 'y_list': [x.tolist() for x in learner.y_list]},outfile)

from . import samplers
@click.command()
@click.argument('example', type = click.Choice(EXAMPLES))
@click.argument('outputfile')
@click.option('--baseline', default = 'latin', type = click.Choice(['latin','grids']))
@click.option('--logfile', default = 'excursion.log')
def baseline(example, outputfile, baseline, logfile):
    setup_logging(logfile)
    oneshot_generator = getattr(samplers, {
        'latin': 'latin_hypercube_generator',
        'grids': 'regular_grid_generator'
    }[baseline])

    oneshot_options = {
        'latin': dict(nsamples_per_npoints=10, point_range=[4,10]),
        'grids': dict(central_range = [5,20], nsamples_per_grid = 10, min_points_per_dim = 2)
    }[baseline]

    example = load_example(example)
    metrics = {'metrics': [], 'X': [], 'y_list': []}

    for X,info in oneshot_generator(example, **oneshot_options):
        start = time.time()
        l = Learner(example)
        y_list  = l.evaluator(l.scandetails,X)
        l.tell(X,y_list)
        delta = time.time()-start
        metrics['X'].append(X.tolist())
        metrics['y_list'].append([y.tolist() for y in y_list])
        metrics['metrics'].append(l.metrics[0])
        msg = 'baseline type {} {} | misclass {:.3E} | npoints {:5.0f} | time {:10.2f} s | {}'.format(
            baseline, info or '',
            1-l.metrics[-1]['confusion']['t'],
            l.metrics[-1]['npoints'],
            delta,
            outputfile
        )
        log.info(msg)
    with open(outputfile,'w') as outfile:
        json.dump(metrics,outfile)
