# -*- coding: utf-8 -*-

import numpy as np
from .utils import PrepareModule


class Solver(object):
    def __init__(self):
        self.Model = None
        self.cache = {}

    def Reset(self):
        self.cache = {}

    def Update(self, layers):
        for layer in layers:
            if layer.LEARNABLE:
                for wIdx, (wName, gradName, applyDecay, *misc) in enumerate(layer.WEIGHT):
                    self.DoUpdate(layer, wIdx, wName, gradName, applyDecay, misc)

    def DoUpdate(self, *args):
        raise NotImplementedError


class SGD(Solver):
    def __init__(self, lr=0.1, decay=0):
        super().__init__()
        self.lr = lr
        self.decay = decay

    def DoUpdate(self, layer, wIdx, wName, gradName, applyDecay, misc):
        w = getattr(layer, wName)
        dw = getattr(layer, gradName)
        lr = self.lr if layer.lr[wIdx] is None else layer.lr[wIdx]
        decay = self.decay if layer.decay[wIdx] is None else layer.decay[wIdx]
        if applyDecay and decay:
            setattr(layer, wName, w - lr * (dw + decay * w))
        else:
            setattr(layer, wName, w - lr * dw)


class Momentum(Solver):
    def __init__(self, lr=0.1, gamma=0.9, decay=0):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.decay = decay

    def DoUpdate(self, layer, wIdx, wName, gradName, applyDecay, misc):
        w = getattr(layer, wName)
        dw = getattr(layer, gradName)
        lr = self.lr if layer.lr[wIdx] is None else layer.lr[wIdx]
        decay = self.decay if layer.decay[wIdx] is None else layer.decay[wIdx]
        if (layer, wName) in self.cache:
            vt_p = self.cache[(layer, wName)]
        else:
            vt_p = np.zeros_like(w)
        if applyDecay and decay:
            vt = self.gamma * vt_p + lr * (dw + decay * w)
        else:
            vt = self.gamma * vt_p + lr * dw
        self.cache[(layer, wName)] = vt
        setattr(layer, wName, w - vt)


class Nesterov(Solver):
    def __init__(self):
        super().__init__()

    def DoUpdate(self, layers):
        pass


class Rprop(Solver):
    def __init__(self, lr=0.01, eta=(), size=()):
        super().__init__()

    def DoUpdate(self, layers):
        pass


# ======================================================================================================================
SOLVER = {
    "SGD"     : SGD,
    "MOMENTUM": Momentum,
}

__all__ = PrepareModule(SOLVER, "SOLVER")
