# -*- coding: utf-8 -*-

# Epoch:     0, Cost: 1.116698
# Epoch:  1000, Cost: 0.068308
# Epoch:  2000, Cost: 0.041813
# Epoch:  3000, Cost: 0.032390
# Epoch:  4000, Cost: 0.027661
# Epoch:  5000, Cost: 0.024789
# Epoch:  6000, Cost: 0.022837
# Epoch:  7000, Cost: 0.021408
# Epoch:  8000, Cost: 0.020304
# Epoch:  9000, Cost: 0.019416
# Epoch: 10000, Cost: 0.018671

import numpy as np
from toydata import load_spiral_dataset, plot_decision_boundary

from simplenn.model import Model
from simplenn.utils import GroundTruth

np.random.seed(1024)

X, y = load_spiral_dataset(3)
gt = GroundTruth(y)
m, n_x = X.shape
n_y = 3
n_h = 20
decay = 0
learning_rate = 1
num_epochs = 10_000

# Use Linear + CE
# model = Model(
#     layers=[
#         ("LINEAR", n_x, n_h),
#         ("RELU",),
#         ("LINEAR", n_h, n_y),
#     ],
#     loss=("CE",),
#     solver=("SGD", learning_rate, decay))

# Or use LOG_SOFTMAX + NLL
model = Model(
    layers=[
        ("LINEAR", n_x, n_h),
        ("RELU",),
        ("LINEAR", n_h, n_y),
        ("LOG_SOFTMAX", n_h, n_y),
    ],
    loss=("NLL",),
    solver=("SGD", learning_rate, decay))

# for epoch in range(1, num_epochs + 1):
#     try:
#         model.TrainStep(X, gt)
#         if epoch % 100 == 0:
#             print(model.cost)
#     except KeyboardInterrupt:
#         print("Keyboard Interrupted")
#         break

# for epoch in range(1, num_epochs // 100 + 1):
#     try:
#         model.Train100Steps(X, gt)
#         print(model.cost)
#     except KeyboardInterrupt:
#         print("Keyboard Interrupted")
#         break

model.Train(X, gt, num_epochs, 1000)

plot_decision_boundary(lambda x: model.Predict(x), X, y)
