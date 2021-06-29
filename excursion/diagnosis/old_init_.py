import numpy as np
import logging

log = logging.getLogger(__name__)

def get_class_labels(y,class_labels,minmaxes):
    assigned_labels = -np.ones_like(y)
    for class_idx in class_labels:
        class_selector = np.logical_and(y>=minmaxes[class_idx][0], y<minmaxes[class_idx][1])
        assigned_labels[class_selector] = class_idx
    assert np.min(assigned_labels) >= 0
    return assigned_labels

def get_status(testy,esti_y,class_labels,minmaxes):
    true_labels = get_class_labels(testy,class_labels,minmaxes)
    esti_labels = get_class_labels(esti_y,class_labels,minmaxes)
    status = np.concatenate([esti_labels.reshape(-1,1),true_labels.reshape(-1,1)], axis=-1)
    return status

def get_confusion_matrices(statuses,class_labels):
    confusion_matrix = -np.ones((len(statuses),2,2))
    for sidx,sta in enumerate(statuses):
        for i in class_labels:
            for j in class_labels:
                test_val  = i
                truth_val = j
                condition_on_truth = sta[sta[:,1]==truth_val]
                is_test_condition_truth = condition_on_truth[condition_on_truth[:,0]==test_val]
                prob = len(is_test_condition_truth)/len(condition_on_truth)
                confusion_matrix[sidx][i][j] = prob #p(i|j)
    return confusion_matrix


def confusion_callback(X,y_list,gps,scandetails):
    testX, testy_list = scandetails.testdata()
    log.debug('computing confusion matrix based on {} testing points'.format(len(testX)))
    estimatey_list = [gps[i].predict(testX) for i in range(len(scandetails.functions))]

    class_labels = np.arange(len(scandetails.thresholds)+1)

    boundaries = [-np.inf] + scandetails.thresholds + [np.inf]
    minmaxes = [[boundaries[i],boundaries[i+1]] for i in class_labels]
    statuses = [
        get_status(yl,estimatey_list[i],class_labels,minmaxes)
        for i,yl in enumerate(testy_list)
    ]
    matrices = get_confusion_matrices(statuses,class_labels)
    topline = np.mean([np.mean(np.diag(cm)) for cm in matrices])
    return {'t': topline, 'matrices': np.asarray(matrices).tolist()}

def diagnose(X,y_list,gps, scandetails, callbacks = None):
    callbacks = callbacks or {
        'confusion': confusion_callback,
        'npoints': lambda X,*args: len(X),
    }
    return {name: c(X,y_list,gps,scandetails) for name,c in callbacks.items()}