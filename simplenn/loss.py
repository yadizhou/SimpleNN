# -*- coding: utf-8 -*-

import numpy as np
from .utils import PrepareModule


# ==============================================================================
class Loss(object):
    def __init__(self):
        pass

    def __call__(self, y, h):
        return self.Forward(y, h), self.Backward(y, h)

    def Forward(self, y, h):
        raise NotImplementedError

    def Backward(self, y, h):
        raise NotImplementedError

    def Reset(self):
        pass


# ==============================================================================
class MSE(Loss):
    def Forward(self, y, h):
        return np.sum((y - h) ** 2) / 2 / y.shape[0]

    def Backward(self, y, h):
        return (h - y) / y.shape[0]


class BCE(Loss):
    def Forward(self, y, h):
        return -np.sum(y * np.log(h) + (1 - y) * np.log1p(-h)) / y.shape[0]

    def Backward(self, y, h):
        return (h - y) / (1 - h) / h / y.shape[0]


class CE(Loss):  # Takes output from Linear layer, not Softmax layer
    def __call__(self, y, dataIn):
        return self.Forward(y, dataIn), self.Backward(y, self.h)

    def Forward(self, gt, dataIn):
        z = dataIn - dataIn.max(axis=1, keepdims=1)
        e = np.exp(z)
        s = e.sum(axis=1, keepdims=1)
        self.h = e / s
        return -np.sum(gt * (z - np.log(s))) / gt.shape[0]

    def Backward(self, gt, h):  # dJ/dZ = h(Z) - gt
        return (h - gt) / gt.shape[0]

    def Reset(self):
        self.h = None


class NLL(Loss):
    def Forward(self, gt, logh):
        return -np.sum(gt * logh) / gt.shape[0]

    def Backward(self, gt, logh):
        h = np.exp(logh)
        return (h - gt) / (1 - h) / gt.shape[0]


# ==============================================================================
LOSS = {
    "MSE": MSE,
    "BCE": BCE,
    "CE" : CE,
    "NLL": NLL,
}

__all__ = PrepareModule(LOSS, "LOSS")
