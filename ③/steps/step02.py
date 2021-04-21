import time
import math
import matplotlib.pyplot as plt
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable, Layer, Model, DataLoader
from dezero.utils import plot_dot_graph
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers
from dezero.models import MLP
import dezero


# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = np.sin(2*np.pi*x)+np.random.rand(100, 1)
lr = 1.0
batch_size = 100
max_epoch = 5
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((hidden_size, 10))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:

        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data)*len(t)
    elapsed_time = time.time()-start
    print("epoch: {}, loss:{:.4f}, time:{:.4f}".format(
        epoch+1, sum_loss/len(train_set), elapsed_time))
