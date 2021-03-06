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
        self.Model = None
        self.cache = None
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


# ========================== Linear ========================== #
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
        self.db = gradIn.sum(axis=0)
        return gradIn @ self.W.T

    def Predict(self, dataOut):
        return dataOut.argmax(1).reshape(-1, 1)


# ======================== Non-linear ======================== #
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


# ========================= Reshape ========================== #
class Reshape(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def Forward(self, dataIn):
        self.cache = dataIn.shape
        return dataIn.reshape(self.shape)

    def Backward(self, dataIn, dataOut, gradIn):
        return gradIn.reshape(self.cache)


# ======================= Convolution ======================== #
# https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
# Very slow
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


# ========================= Pooling ========================== #
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


# ========================= Dropout ========================== #
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def Forward(self, dataIn):
        if self.Model is None or self.Model.training:
            self.cache = (np.random.rand(*dataIn.shape) < self.p) / self.p
            return dataIn * self.cache
        else:
            self.cache = None
            return dataIn

    def Backward(self, dataIn, dataOut, gradIn):
        if self.Model is None or self.Model.training:
            return gradIn * self.cache
        else:
            return gradIn


# ======================== BatchNorm ========================= #
class BatchNorm(Layer):
    LEARNABLE = True
    WEIGHT = ("g", "dg", DECAY_N, 1), \
             ("b", "db", DECAY_N, 0)

    def __init__(self, nIn, eps=1e-8, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.D = nIn
        self.eps = eps
        self.momentum = momentum
        self.g = Initialize((nIn,), self.init[0])
        self.b = Initialize((nIn,), self.init[1])
        self.dg = None
        self.db = None
        self.cache = {"runM": np.zeros(self.D), "runV": np.zeros(self.D), "----": None}

    def Reset(self):
        self.cache = {"runM": np.zeros(self.D), "runV": np.zeros(self.D), "----": None}

    def Forward(self, dataIn):
        if self.Model is None or self.Model.training:
            m = dataIn.mean(axis=0)
            v = dataIn.var(axis=0)
            v_eps = v + self.eps
            std = np.sqrt(v_eps)
            x_mu = dataIn - m
            x_norm = x_mu / std
            self.cache["runM"] = self.momentum * self.cache["runM"] + (1.0 - self.momentum) * m
            self.cache["runV"] = self.momentum * self.cache["runV"] + (1.0 - self.momentum) * v
            self.cache["----"] = (m, v, v_eps, std, x_mu, x_norm)
            return self.g * x_norm + self.b
        else:
            return self.g * (dataIn - self.cache["runM"]) / np.sqrt(self.cache["runV"] + self.eps) + self.b

    def Backward(self, dataIn, dataOut, gradIn):
        m, v, v_eps, std, x_mu, x_norm = self.cache["----"]
        self.dg = np.sum(gradIn * x_norm, axis=0)
        self.db = np.sum(gradIn, axis=0)
        N, D = dataIn.shape
        gradOut = (self.g / N) / std * (N * gradIn - np.sum(gradIn, axis=0) - x_mu / v_eps * np.sum(gradIn * x_mu, axis=0))
        return gradOut


# =========================== RNN ============================ #
class RNN(Layer):
    LEARNABLE = True
    WEIGHT = ("wi", "dwi", DECAY_Y, "XAVIER_UNIFORM"), \
             ("bi", "dbi", DECAY_N, 0), \
             ("wh", "dwh", DECAY_Y, "XAVIER_UNIFORM"), \
             ("bh", "dbh", DECAY_N, 0),

    def __init__(self, nIn, nOut, nLayer, func="tanh", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nIn = nIn
        self.nOut = nOut
        self.nLayer = nLayer
        self.func = func
        # self.wi = Initialize((,), self.init[0])
        # self.bi = Initialize((,), self.init[1])
        # self.wh = Initialize((,), self.init[2])
        # self.bh = Initialize((,), self.init[3])
        self.dwi = self.dbi = self.dwh = self.dbh = None

    def Forward(self, dataIn):
        pass

    def Backward(self, dataIn, dataOut, gradIn):
        pass


# =========================== LSTM =========================== #


# =========================== GRU ============================ #


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

    "DROPOUT"       : Dropout,
    "BATCH_NORM"    : BatchNorm,

}

__all__ = PrepareModule(LAYER, "LAYER")
