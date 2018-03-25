# -*- coding: utf-8 -*-

import math
import numpy as np
from .utils import PrepareModule


# ==============================================================================
def Uniform(shape, a=0.0, b=1.0):
    return np.random.uniform(low=a, high=b, size=shape)


def Normal(shape, mean=0.0, std=1.0):
    return np.random.normal(loc=mean, scale=std, size=shape)


def Constant(shape, value):
    return np.full(shape=shape, fill_value=value)


def Eye(shape):
    return np.eye(N=shape[0], M=shape[1])


def XavierUniform(shape, gain=1.0):
    a = gain * math.sqrt(6) / math.sqrt(sum(shape))
    return Uniform(shape, a=-a, b=a)


def XavierNormal(shape, gain=1.0):
    std = gain * math.sqrt(2) / math.sqrt(sum(shape))
    return Normal(shape, std=std)


# ==============================================================================
INIT = {
    "UNIFORM"       : Uniform,
    "NORMAL"        : Normal,
    "CONSTANT"      : Constant,
    "EYE"           : Eye,
    "XAVIER_UNIFORM": XavierUniform,
    "XAVIER_NORMAL" : XavierNormal,
}

__all__ = PrepareModule(INIT, "INIT", "Initialize")


def Initialize(shape, method):
    if isinstance(method, (int, float)):
        return Constant(shape, method)
    if isinstance(method, (tuple, list)):
        method, *args = method
    else:
        args = ()
    return INIT[method](shape, *args)
