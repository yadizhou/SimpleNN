# -*- coding: utf-8 -*-

from .utils import PrepareModule


class Solver(object):
    def __init__(self):
        self.cache = None

    def Reset(self):
        self.cache = None

    def Update(self, layers):
        raise NotImplementedError


class SGD(Solver):
    def __init__(self, lr=0.1, decay=0):
        super().__init__()
        self.lr = lr
        self.decay = decay

    def Update(self, layers):
        for layer in layers:
            if layer.LEARNABLE:
                for wIdx, (wName, gradName, applyDecay, _) in enumerate(layer.WEIGHT):
                    w = getattr(layer, wName)
                    dw = getattr(layer, gradName)
                    lr = self.lr if layer.lr[wIdx] is None else layer.lr[wIdx]
                    decay = self.decay if layer.decay[wIdx] is None else layer.decay[wIdx]
                    if applyDecay and decay:
                        setattr(layer, wName, w - lr * (dw + decay * w))
                    else:
                        setattr(layer, wName, w - lr * dw)


# ======================================================================================================================
SOLVER = {
    "SGD": SGD,
}

__all__ = PrepareModule(SOLVER, "SOLVER")
