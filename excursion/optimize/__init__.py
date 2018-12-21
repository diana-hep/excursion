import numpy as np
import logging
import time
import os
from . import core
from .gaussian_process import get_gp

log = logging.getLogger(__name__)

def run_all_acpoints(acqX, gps, thresholds, meanX):
    try:
        from joblib import Parallel, delayed
        nparallel = int(os.environ.get('EXCURSION_NPARALLEL',os.cpu_count()))
        log.debug('analyzing {} candidate acquisition points in {} parallel workers'.format(
            len(acqX),nparallel)
        )
        start = time.time()
        result = Parallel(nparallel)(
            delayed(core.info_gain)(xtest, gps, thresholds, meanX) for xtest in acqX
        )
        delta = time.time()-start
        log.debug('acquisition analysis took in {:.3f} seconds'.format(delta))
        return np.asarray(result)
    except ImportError:
        log.debug('joblib not found. falling back to serial')
        return np.array([
            core.info_gain(xtest, gps, thresholds, meanX) for xtest in acqX]
        )

def _search_single(gps, X, scandetails):
    acqX  = scandetails.acqX()
    meanX = scandetails.meanX()
    thresholds = [-np.inf] + scandetails.thresholds + [np.inf]
    acqval     = run_all_acpoints(acqX, gps, thresholds, meanX)
    
    newx = None
    for i,cacq in enumerate(acqX[np.argsort(acqval)]):
        if cacq.tolist() not in X.tolist():
            log.debug('taking new x. the best non-existent index {} {}'.format(i,cacq))
            newx = cacq
            return newx,acqX,acqval
        else:
            log.debug('{} is good but already there'.format(cacq))
    log.warning('returning None.. something must be wrong')
    return None,None

def default_evaluator(scandetails,newX):
    return [func(newX) for func in scandetails.functions]

def init(
    scandetails, n_init = 5, seed = None,
    evaluator = default_evaluator, gp_maker = get_gp
    ):
    X = scandetails.random_points(n_init, seed = seed)
    y_list = evaluator(scandetails,X)
    gps = [gp_maker(X,yl) for yl in y_list]
    return X,y_list,gps

def tell(X, y_list, scandetails, newX, newys_list, gp_maker):
    for i,newys in enumerate(newys_list):
        log.debug('Evaluted function {} to values: {}'.format(i,newys))
        y_list[i] = np.concatenate([y_list[i],newys])
    X = np.concatenate([X,newX])
    gps = [gp_maker(X,y_list[i]) for i in range(len(scandetails.functions))]
    return X,y_list,gps

def evaluate_and_refine(
    X, y_list, newX, scandetails,
    evaluator = default_evaluator, gp_maker = get_gp
    ):
    newys_list = evaluator(scandetails,newX)
    X, y_list, gps = tell(X, y_list, scandetails, newX, newys_list, gp_maker)
    return X, y_list, gps

def suggest(
    gps, X, scandetails,
    gp_maker = get_gp, batchsize=1,
    resampling_frac = 0.30, return_acqvals = False
    ):
    if batchsize > 1 and not gp_maker:
        raise RuntimeError('need a gp maker for batched acq')
    resample = int(batchsize * resampling_frac)
    log.debug('resample up to %s', resample)
    newX = np.empty((0,X.shape[-1]))
    my_gps    = gps
    orig_gps    = gps
    my_y_list = [np.copy(gp.y_train_) for gp in my_gps]
    myX       = np.copy(X)

    acqinfos = []
    n_orig  = myX.shape[0]

    log.debug('base X is %s',myX.shape)

    while True:
        newx,acqX,acqinfo = _search_single(my_gps, myX, scandetails)
        newX = np.concatenate([newX,np.asarray([newx])])
        myX  = np.concatenate([myX,np.asarray([newx])])
        acqinfos.append({'acqX': acqX, 'acqinfo': acqinfo})
        if(len(newX)) == batchsize:
            log.debug('we got our batch')
            if return_acqvals:
                return newX, acqinfos
            return newX
        log.debug('do the fake update on %s %s',myX.shape,newX.shape)
        newy_list = [gp.sample_y([newx], n_samples = 1)[:,0] for gp in orig_gps]

        resample = min(len(newX),resample)
        for i,newy in enumerate(newy_list):
            log.debug('new y i: {} {}'.format(i,newy))
            my_y_list[i] = np.concatenate([my_y_list[i],newy])

            if resample:
                log.debug('resampling %s %s', resample, np.arange(n_orig,len(myX)))
                new_indices = np.random.choice(
                    np.arange(n_orig,len(myX)), resample, replace = False
                )
                log.debug('indices %s',new_indices)
                resampleX = myX[new_indices]
                log.debug('resampling shape %s',resampleX.shape)
                my_y_list[i][new_indices] = gps[i].sample_y(resampleX, n_samples = 1)[:,0]

            log.debug(my_y_list[i].shape)
        log.debug('build fake gps')
        my_gps = [gp_maker(myX,my_y) for my_y in my_y_list]

