import numpy as np
import itertools
from .. import utils

def classlabels(values,thresholds):
    labels = np.zeros(values.shape)
    for j in range(len(thresholds) - 1):
        print('[{}, {}]'.format(thresholds[j], thresholds[j+1]))
        within = np.logical_and(thresholds[j] < values, values < thresholds[j+1])
        labels[within] = j+1
    return labels

def confusion_matrix(gps, scandetails):
    thresholds = np.concatenate([[-np.inf],scandetails.thresholds,[np.inf]])
    diagX = utils.mesh2points(utils.mgrid(scandetails.plot_rangedef),scandetails.plot_rangedef[:,2])

    diagX = diagX[~scandetails.invalid_region(diagX)]

    labels = list(range(1,len(thresholds)))

    confusion_list, predlabels_list, truelabels_list= [], [], []
    for i,gp in enumerate(gps):
        predy  = gps[i].predict(diagX)
        truthy = scandetails.truth_functions[i](diagX)
        predlabels = classlabels(predy,thresholds)
        truelabels = classlabels(truthy,thresholds)

        confusion_matrix = np.zeros((len(labels),len(labels)))
        for pred,true in itertools.product(labels,labels):
            print('pred {}/true {}'.format(pred,true))
            predlabels_when_true = predlabels[truelabels==true]
            numerator = np.sum(np.where(predlabels_when_true==pred,1,0))
            denominator = len(predlabels_when_true)
            print('{}/{}'.format(numerator,denominator))
            confusion_matrix[true-1,pred-1] = numerator/denominator

        predlabels_list.append(predlabels)
        truelabels_list.append(truelabels)
        confusion_list.append(confusion_matrix.tolist())
    return confusion_list, predlabels_list, truelabels_list, diagX, labels
