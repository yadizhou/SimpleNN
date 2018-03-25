# -*- coding: utf-8 -*-

import numpy as np
from .init import Initialize
from .utils import PrepareModule, ExpectTuple, ExpectTupleInTuple

# ==============================================================================
DECAY_Y = True
DECAY_N = False


# ==============================================================================
class Layer(object):
    LEARNABLE = False
    WEIGHT = ()

    __INITIALIZED__ = False

    def __new__(cls, *args, **kwargs):
        if not cls.__INITIALIZED__:
            if cls.LEARNABLE and not isinstance(cls.WEIGHT[0], (tuple, list)):
                cls.WEIGHT = (cls.WEIGHT,)
            cls.__INITIALIZED__ = True
        return super().__new__(cls)

    def __init__(self, init=None, lr=None, decay=None):
        self.cache = None
        # self.input = None
        # self.output = None
        if self.LEARNABLE:
            self.init = [i[3] for i in self.WEIGHT] if init is None else ExpectTupleInTuple(init, len(self.WEIGHT))
            self.lr = ExpectTuple(lr, len(self.WEIGHT))
            self.decay = ExpectTuple(decay, len(self.WEIGHT))

    def Forward(self, dataIn):
        raise NotImplementedError

    def Backward(self, dataIn, dataOut, grad):
        raise NotImplementedError

    def Predict(self, dataOut):
        raise Exception('Prediction is undefined for layer "%s"' % self.__class__.__name__)

    def Reset(self):
        self.cache = None
        # self.input = None
        # self.output = None


# Linear
class Linear(Layer):
    LEARNABLE = True
    WEIGHT = ("W", "dW", DECAY_Y, "NORMAL"), \
             ("b", "db", DECAY_N, 0)

    def __init__(self, nIn, nOut, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nIn = nIn
        self.nOut = nOut
        self.W = Initialize((nIn, nOut), self.init[0])
        self.b = Initialize((1, nOut), self.init[1])
        self.dW = None
        self.db = None

    def Forward(self, dataIn):
        return dataIn @ self.W + self.b

    def Backward(self, dataIn, dataOut, grad):
        self.dW = dataIn.T @ grad
        self.db = grad.mean(axis=0)
        return grad @ self.W.T

    def Predict(self, dataOut):
        return dataOut.argmax(1).reshape(-1, 1)


# Non-linear
class ReLU(Layer):
    def Forward(self, dataIn):
        return np.maximum(0, dataIn)

    def Backward(self, dataIn, dataOut, grad):
        return grad * (dataIn > 0)


class Sigmoid(Layer):
    def Forward(self, dataIn):
        return 1.0 / (1.0 + np.exp(-dataIn))

    def Backward(self, dataIn, dataOut, grad):
        return grad * dataOut * (1 - dataOut)

    def Predict(self, dataOut):
        return (dataOut > 0.5).reshape(-1, 1)


class Tanh(Layer):
    def Forward(self, dataIn):
        return np.tanh(dataIn)

    def Backward(self, dataIn, dataOut, grad):
        return grad * (1 - dataOut ** 2)


class Softmax(Layer):
    def Forward(self, dataIn):
        e = np.exp(dataIn - dataIn.max(axis=1, keepdims=1))
        return e / e.sum(axis=1, keepdims=1)

    def Backward(self, dataIn, dataOut, grad):
        return grad * dataOut * (1 - dataOut)

    def Predict(self, dataOut):
        return dataOut.argmax(1).reshape(-1, 1)


class LogSoftmax(Layer):
    def Forward(self, dataIn):
        z = dataIn - dataIn.max(axis=1, keepdims=1)
        e = np.exp(z)
        return z - np.log(e.sum(axis=1, keepdims=1))

    def Backward(self, dataIn, dataOut, grad):
        return -grad * np.expm1(dataOut)

    def Predict(self, dataOut):
        return dataOut.argmax(1).reshape(-1, 1)


# ==============================================================================
LAYER = {
    "LINEAR"     : Linear,

    "RELU"       : ReLU,
    "SIGMOID"    : Sigmoid,
    "TANH"       : Tanh,

    "SOFTMAX"    : Softmax,
    "LOG_SOFTMAX": LogSoftmax,

}

__all__ = PrepareModule(LAYER, "LAYER")
