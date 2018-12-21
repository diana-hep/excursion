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
@click.argument('example')
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
        json.dump({'metrics': learner.metrics, 'X': learner.X.tolist(), 'y_list': [x.tolist() for x in learner.y_list]},open(outputfile,'w'))
    # gpopts  = yaml.load(gpopts)
    # acqopts = yaml.load(acqopts)
    # log.debug(gpopts,acqopts)
    # for i,(gps, acqinfo, r) in enumerate(runloop(ninit, scandetails, nupdates, acq_optimizer=acqtype, gpopts = gpopts, acqopts = acqopts)):
    #     log.debug('dumping iteration {}'.format(i))
    #     json.dump(r,open(outputfile,'w'))
