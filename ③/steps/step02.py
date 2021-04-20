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
batch_size = 30
max_epoch = 300
hidden_size = 10

train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)
model = MLP((hidden_size, 10))
optimizer = optimizers.Momentum(lr)
optimizer.setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size/batch_size)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:

        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data)*len(t)
        sum_acc += float(acc.data)*len(t)

    print("epoch:{}".format(epoch+1))
    print("train loss:{:.4f}, accuracy:{:.4f}".format(
        sum_loss/len(train_set), sum_acc/len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in train_loader:

            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data)*len(t)
            sum_acc += float(acc.data)*len(t)
    print("test loss:{:.4f}, accuracy:{:.4f}".format(
        sum_loss/len(test_set), sum_acc/len(test_set)))
    print()
