# -*- coding: utf-8 -*-

import numpy as np


def PrepareModule(d, *ns):
    __all__ = list(ns) + [d[i].__name__ for i in d]
    for key in list(d.keys()):
        d[key.lower()] = d[key]
    return __all__


def ExpectTuple(x, n=1):
    if not isinstance(x, (tuple, list)):
        return (x,) * n
    return x


def ExpectTupleInTuple(x, n=1):
    if not isinstance(x, (tuple, list)):
        return ((x,),) * n
    if not isinstance(x[0], (tuple, list)):
        return (x,) * n
    return x


def GroundTruth(y, nC=None):
    m = y.size
    if nC is None:
        nC = len(np.unique(y))
    gt = np.zeros((m, nC))
    gt[np.arange(m), y.flatten()] = 1
    return gt
