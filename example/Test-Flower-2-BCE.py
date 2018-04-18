# -*- coding: utf-8 -*-

# Epoch:     0, Cost: 0.716379
# Epoch:  1000, Cost: 0.049936
# Epoch:  2000, Cost: 0.026283
# Epoch:  3000, Cost: 0.019794
# Epoch:  4000, Cost: 0.016638
# Epoch:  5000, Cost: 0.014784
# Epoch:  6000, Cost: 0.013544
# Epoch:  7000, Cost: 0.012666
# Epoch:  8000, Cost: 0.012020
# Epoch:  9000, Cost: 0.011491
# Epoch: 10000, Cost: 0.011079

import numpy as np
from toydata import load_spiral_dataset, plot_decision_boundary

from simplenn.model import Model

np.random.seed(1024)

X, y = load_spiral_dataset(2)

m, n_x = X.shape
n_y = 1
n_h = 20
decay = 0
learning_rate = 1
num_epochs = 10_000

model = Model(
    layers=[
        ("LINEAR", n_x, n_h),
        ("RELU",),
        ("LINEAR", n_h, n_y),
        ("SIGMOID",)
    ],
    loss=("BCE",),
    solver=("MOMENTUM", learning_rate, 0.5, decay))
    # solver=("SGD", learning_rate, decay))

## Alternatively,
# from simplenn.layer import Linear, ReLU, Sigmoid
# from simplenn.loss import BCE
# from simplenn.solver import SGD
# model = Model(
#     layers=[Linear(n_x, n_h),
#             ReLU(),
#             Linear(n_h, n_y),
#             Sigmoid()],
#     loss=BCE(),
#     solver=SGD(lr=learning_rate, decay=decay))

model.Train(X, y, num_epochs, 1000)

plot_decision_boundary(lambda x: model.Predict(x), X, y)
