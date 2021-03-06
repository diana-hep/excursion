import numpy as np
import logging
from . import core
from .. import get_gp

log = logging.getLogger(__name__)

def _gridsearch(gps, X, scandetails):
    thresholds = [-np.inf] + scandetails.thresholds + [np.inf]
    acqval     = np.array([
        core.info_gain(xtest, gps, thresholds, scandetails.meanX) for xtest in scandetails.acqX]
    )
    
    newx = None
    for i,cacq in enumerate(scandetails.acqX[np.argsort(acqval)]):
        if cacq.tolist() not in X.tolist():
            log.info('taking new x. the best non-existent index {} {}'.format(i,cacq))
            newx = cacq
            return newx,acqval
        else:
            log.info('{} is good but already there'.format(cacq))
    log.warning('returning None.. something must be wrong')
    return None,None

def init(scandetails, n_init = 5, seed = None, gp_maker = get_gp):
    ndim = scandetails.acqX.shape[-1]
    nfuncs = len(scandetails.truth_functions)
    np.random.seed(seed)
    X = np.random.uniform(scandetails.plot_rangedef[:,0],scandetails.plot_rangedef[:,1], size = (n_init,ndim))
    y_list = [np.array([scandetails.truth_functions[i](np.asarray([x]))[0] for x in X]) for i in range(nfuncs)]
    gps = [get_gp(X,y_list[i]) for i in range(nfuncs)]
    return X,y_list,gps

def gridsearch(
    gps, X, scandetails,
    gp_maker = get_gp, batchsize=1,
    resampling_frac = 0.30
    ):
    if batchsize > 1 and not gp_maker:
        raise RuntimeError('need a gp maker for batched acq')
    resample = int(batchsize * resampling_frac)
    log.info('resample up to %s', resample)
    newX = np.empty((0,X.shape[-1]))
    my_gps    = gps
    orig_gps    = gps
    my_y_list = [np.copy(gp.y_train_) for gp in my_gps]
    myX       = np.copy(X)

    acqinfos = []
    n_orig  = myX.shape[0]

    log.info('base X is %s',myX.shape)

    while True:
        newx,acqinfo = _gridsearch(my_gps, myX, scandetails)
        newX = np.concatenate([newX,np.asarray([newx])])
        myX  = np.concatenate([myX,np.asarray([newx])])
        acqinfos.append(acqinfo)
        if(len(newX)) == batchsize:
            log.info('we got our batch')
            return newX, acqinfos
        log.info('do the fake update on %s %s',myX.shape,newX.shape)
        # newy_list = [gp.predict(newX) for gp in my_gps]
        newy_list = [gp.sample_y([newx], n_samples = 1)[:,0] for gp in orig_gps]

        resample = min(len(newX),resample)
        for i,newy in enumerate(newy_list):
            log.info('new y i: {} {}'.format(i,newy))
            my_y_list[i] = np.concatenate([my_y_list[i],newy])

            if resample:
                log.info('resampling %s %s', resample, np.arange(n_orig,len(myX)))
                new_indices = np.random.choice(
                    np.arange(n_orig,len(myX)), resample, replace = False
                )
                log.info('indices %s',new_indices)
                resampleX = myX[new_indices]
                log.info('resampling shape %s',resampleX.shape)
                my_y_list[i][new_indices] = gps[i].sample_y(resampleX, n_samples = 1)[:,0]

            log.info(my_y_list[i].shape)
        log.info('build fake gps')
        my_gps = [gp_maker(myX,my_y) for my_y in my_y_list]
