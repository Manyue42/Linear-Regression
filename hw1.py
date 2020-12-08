# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy

csv = 'https://www.dropbox.com/s/0rjqoaygjbk3sp8/boston_house_prices_3features.txt?dl=1'
data = numpy.genfromtxt(csv, delimiter=',', skip_header=1)

import torch

inputs = data[:, [0,1,2]]
inputs = inputs.astype(numpy.float32)
inputs = torch.from_numpy(inputs)

target = data[:,3]
target = target.astype(numpy.float32)
target = torch.from_numpy(target)

# set variables
learning_rate = 0.002
msels=[]

#weights and biases
w= torch.randn(1, 3, requires_grad = True)
b = torch.randn(1, requires_grad = True)
print(w)
print(b)

# Define the model
def model(x):
    return x @ w.t() + b

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

for i in range(200):
    print("Epoch", i, ":")
    preds = model(inputs)
    loss = mse(preds, target)
    loss.backward()
    print("Loss=", loss)
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        w.grad.zero_()
        b.grad.zero_()
    
    msels.append(loss)
    
plt.plot(numpy.linspace(0,200,200),msels)