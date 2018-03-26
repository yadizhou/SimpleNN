# -*- coding: utf-8 -*-

import numpy as np
from .init import Initialize
from .utils import PrepareModule, ExpectTuple, ExpectTupleInTuple
from .im2col import col2im_indices, im2col_indices

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

    def Backward(self, dataIn, dataOut, gradIn):
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
    WEIGHT = ("W", "dW", DECAY_Y, "XAVIER_UNIFORM"), \
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

    def Backward(self, dataIn, dataOut, gradIn):
        self.dW = dataIn.T @ gradIn
        self.db = gradIn.mean(axis=0)
        return gradIn @ self.W.T

    def Predict(self, dataOut):
        return dataOut.argmax(1).reshape(-1, 1)


# Non-linear
class ReLU(Layer):
    def Forward(self, dataIn):
        return np.maximum(0, dataIn)

    def Backward(self, dataIn, dataOut, gradIn):
        return gradIn * (dataIn > 0)


class Sigmoid(Layer):
    def Forward(self, dataIn):
        return 1.0 / (1.0 + np.exp(-dataIn))

    def Backward(self, dataIn, dataOut, gradIn):
        return gradIn * dataOut * (1 - dataOut)

    def Predict(self, dataOut):
        return (dataOut > 0.5).reshape(-1, 1)


class Tanh(Layer):
    def Forward(self, dataIn):
        return np.tanh(dataIn)

    def Backward(self, dataIn, dataOut, gradIn):
        return gradIn * (1 - dataOut ** 2)


class Softmax(Layer):
    def Forward(self, dataIn):
        e = np.exp(dataIn - dataIn.max(axis=1, keepdims=1))
        return e / e.sum(axis=1, keepdims=1)

    def Backward(self, dataIn, dataOut, gradIn):
        return gradIn * dataOut * (1 - dataOut)

    def Predict(self, dataOut):
        return dataOut.argmax(1).reshape(-1, 1)


class LogSoftmax(Layer):
    def Forward(self, dataIn):
        z = dataIn - dataIn.max(axis=1, keepdims=1)
        e = np.exp(z)
        return z - np.log(e.sum(axis=1, keepdims=1))

    def Backward(self, dataIn, dataOut, gradIn):
        return -gradIn * np.expm1(dataOut)

    def Predict(self, dataOut):
        return dataOut.argmax(1).reshape(-1, 1)


# Reshape
class Reshape(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def Forward(self, dataIn):
        self.cache = dataIn.shape
        return dataIn.reshape(self.shape)

    def Backward(self, dataIn, dataOut, gradIn):
        return gradIn.reshape(self.cache)


# Convolution
# Adapted from https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
# Extremely slow
class Conv2d(Layer):
    LEARNABLE = True
    WEIGHT = ("W", "dW", True, "XAVIER_UNIFORM"), \
             ("b", "db", False, 0)

    def __init__(self, cIn, cOut, field, stride=1, padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = cIn
        self.D = cOut
        self.field = ExpectTuple(field, 2)
        self.stride = stride
        self.padding = padding
        self.W = Initialize((cOut, cIn, *self.field), self.init[0])
        self.b = Initialize((cOut, 1), self.init[1])
        self.dW = None
        self.db = None

    def Forward(self, dataIn):
        n_x, d_x, h_x, w_x = dataIn.shape
        h_out = int((h_x - self.field[0] + 2 * self.padding) / self.stride + 1)
        w_out = int((w_x - self.field[1] + 2 * self.padding) / self.stride + 1)
        X_col = im2col_indices(dataIn, self.field[0], self.field[1], padding=self.padding, stride=self.stride)
        W_col = self.W.reshape(self.D, -1)
        out = W_col @ X_col + self.b
        out = out.reshape(self.D, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        self.cache = X_col
        return out

    def Backward(self, dataIn, dataOut, back):
        X_col = self.cache
        n_filter, d_filter, h_filter, w_filter = self.W.shape
        db = np.sum(back, axis=(0, 2, 3))
        self.db = db.reshape(n_filter, -1)
        dout_reshaped = back.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        self.dW = dW.reshape(self.W.shape)
        W_reshape = self.W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dout_reshaped
        back = col2im_indices(dX_col, dataIn.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)
        return back


# Pooling
class MaxPooling2d(Layer):
    def __init__(self, field=2, stride=2):
        super().__init__()
        self.field = field
        self.stride = stride

    def Forward(self, dataIn):
        N, C, H, W = dataIn.shape
        H_O = int((H - self.field) / self.stride + 1)
        W_O = int((W - self.field) / self.stride + 1)
        X_reshaped = dataIn.reshape(N * C, 1, H, W)
        X_col = im2col_indices(X_reshaped, self.field, self.field, padding=0, stride=self.stride)
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        out = out.reshape(H_O, W_O, N, C)
        out = out.transpose(2, 3, 0, 1)
        self.cache = (X_col, max_idx)
        return out

    def Backward(self, dataIn, dataOut, gradIn):
        N, C, H, W = dataIn.shape
        X_col, max_idx = self.cache
        dX_col = np.zeros_like(X_col)
        dout_flat = gradIn.transpose(2, 3, 0, 1).ravel()
        dX_col[max_idx, range(max_idx.size)] = dout_flat
        dX = col2im_indices(dX_col, (N * C, 1, H, W), self.field, self.field, padding=0, stride=self.stride)
        dX = dX.reshape(dataIn.shape)
        return dX


# ==============================================================================
LAYER = {
    "LINEAR"        : Linear,

    "RELU"          : ReLU,
    "SIGMOID"       : Sigmoid,
    "TANH"          : Tanh,
    "SOFTMAX"       : Softmax,
    "LOG_SOFTMAX"   : LogSoftmax,

    "RESHAPE"       : Reshape,
    "CONV_2D"       : Conv2d,
    "MAX_POOLING_2D": MaxPooling2d,

}

__all__ = PrepareModule(LAYER, "LAYER")
