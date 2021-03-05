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
        try:
            newX =  learner.suggest(batchsize=n_batch)
            learner.evaluate_and_tell(newX)
            yield learner
        except:
            log.warning('suggestion failed. ending generator.')
            return

def load_learner(example, inputfile = None, init_config = None):
    if not inputfile:
        n_init = init_config['ninit']
        log.debug('initializing')
        assert n_init
        learner = Learner(load_example(example))
        learner.initialize(n_init = n_init)
    else:
        log.info('load snapshot')
        intputdata = json.load(open(inputfile))
        learner = Learner(load_example(example))
        learner.initialize(snapshot = intputdata)

    learner.scandetails._nacq = init_config['nacq']
    learner.scandetails._nmean = init_config['nmean']
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

import signal
INTERRUPT = False
def signal_handler(sig, frame):
    global INTERRUPT
    log.warning('SIG_INT caught. Interrupting at next possible opportunity')
    INTERRUPT = True
signal.signal(signal.SIGINT, signal_handler)


@click.command()
@click.argument('example', type = click.Choice(EXAMPLES))
@click.argument('outputfile')
@click.option('--ninit', default = 10)
@click.option('--nacq', default = 1000)
@click.option('--nmean', default = 1000)
@click.option('--nbatch', default = 1)
@click.option('--nupdates', default = 100)
@click.option('--logfile', default = 'excursion.log')
@click.option('--snapshot', default = None)
def main(example, outputfile, logfile, ninit, nacq, nmean, nbatch, nupdates, snapshot):
    global INTERRUPT
    setup_logging(logfile)

    learneropts = {'ninit': ninit, 'nacq': nacq, 'nmean': nmean}

    learner = load_learner(example, inputfile = snapshot, init_config = learneropts)
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
        log.debug('dump to: %s', outputfile)
        with open(outputfile,'w') as outfile:
            json.dump({'metrics': learner.metrics, 'X': learner.X.tolist(), 'y_list': [x.tolist() for x in learner.y_list]},outfile)
        log.debug('interrupt?: %s', INTERRUPT)
        if INTERRUPT:
            log.warning('run was interrupted. exiting gracefully')
            break

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

    example = load_example(example)
    metrics = {'metrics': [], 'X': [], 'y_list': []}

    oneshot_options = {
        'latin': dict(nsamples_per_npoints=1, point_range=[2**example.ndim,11**example.ndim]),
        'grids': dict(central_range = [5,15], nsamples_per_grid = 1, min_points_per_dim = 2)
    }[baseline]

    log.info(oneshot_options)

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
