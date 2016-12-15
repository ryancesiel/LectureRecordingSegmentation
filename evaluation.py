import re
import math
import numpy


def evaluate(boundary, target_boundry, err_tolerance = 2):
    tp = 0
    fp = 0
    fn = 0
    for i in range(1, len(target_boundry)):
        if target_boundry[i] != 0:
            flag = False
            for j in range(i-err_tolerance, i+err_tolerance):
                if boundary[j] != 0:
                    flag = True
                    break
            if flag:
                tp+=1
            else:
                fn+=1
    total = sum([0 if boundary[i]==0 else 1 for i in range(len(boundary))])
    total2 = sum([0 if target_boundry[i]==0 else 1 for i in range(len(target_boundry))])
    fp = total - tp
    # print (total, total2, tp, fp, fn)
    precision = 1.0 * tp / (fp + tp)
    recall = 1.0 * tp / (fn + tp)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

def baseline(target_boundry):
    precisions = []
    recalls = []
    f1s = []
    boundary_len = len(target_boundry)
    num_of_boundry = sum([0 if target_boundry[i]==0 else 1 for i in range(len(target_boundry))])

    for i in range(1000):
        baseline = numpy.random.choice(len(target_boundry), num_of_boundry)
        # print (baseline)
        baseline_boundry = [0] * boundary_len
        for b in baseline:
            baseline_boundry[b] = 1
        precision, recall, f1 = evaluate(baseline_boundry,target_boundry)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    print (sum(precisions)/ len(precisions), sum(recalls)/ len(recalls), sum(f1s)/ len(f1s))
    return sum(precisions)/ len(precisions), sum(recalls)/ len(recalls), sum(f1s)/ len(f1s)

