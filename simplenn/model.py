# -*- coding: utf-8 -*-

from .layer import LAYER
from .loss import LOSS
from .solver import SOLVER


class Model(object):
    def __init__(self, layers, loss, solver):
        self.SetLayers(layers)
        self.SetLoss(loss)
        self.SetSolver(solver)
        self.cost = None
        self.epochsTrained = 0

    def Reset(self):
        self.cost = None
        self.epochsTrained = 0
        for layer in self.Layers:
            layer.Reset()
        self.Loss.Reset()
        self.Solver.Reset()

    def SetLayers(self, layers):
        self.Layers = [LAYER[layer[0]](*layer[1:]) if isinstance(layer, (list, tuple)) else layer for layer in layers]

    def SetLoss(self, loss):
        self.Loss = LOSS[loss[0]](*loss[1:]) if isinstance(loss, (list, tuple)) else loss

    def SetSolver(self, solver):
        self.Solver = SOLVER[solver[0]](*solver[1:]) if isinstance(solver, (list, tuple)) else solver

    def GetLayers(self):
        return self.Layers

    def GetLoss(self):
        return self.Loss

    def GetSolver(self):
        return self.Solver

    def TrainStep(self, x, y):
        cache = self.Forward(x)
        self.cost, grad = self.Loss(y, cache[-1])
        self.Backward(cache, grad)
        self.Solver.Update(self.Layers)
        self.epochsTrained += 1

    def Train10Steps(self, x, y):
        for step in range(10):
            self.TrainStep(x, y)

    def Train100Steps(self, x, y):
        for step in range(100):
            self.TrainStep(x, y)

    def Train1000Steps(self, x, y):
        for step in range(1000):
            self.TrainStep(x, y)

    def Train(self, x, y, steps=100, showCost=1):
        template = "Epoch: %{}d, Cost: %.6f".format(len(str(steps)))
        if showCost:
            for step in range(steps):
                self.TrainStep(x, y)
                if step % showCost == 0:
                    print(template % (step, self.cost))
            self.cost, _ = self.Loss(y, self.Predict(x, prob=True))
            print(template % (steps, self.cost))
        else:
            for step in range(steps):
                self.TrainStep(x, y)

    def Predict(self, x, prob=False):
        for layer in self.Layers:
            x = layer.Forward(x)
        if prob:
            return x
        else:
            return self.Layers[-1].Predict(x)

    def Forward(self, x):
        cache = [x]
        for layer in self.Layers:
            cache.append(layer.Forward(cache[-1]))
        return cache

    def Backward(self, cache, grad):
        for dataIn, dataOut, layer in zip(reversed(cache[:-1]), reversed(cache[1:]), reversed(self.Layers)):
            grad = layer.Backward(dataIn, dataOut, grad)
