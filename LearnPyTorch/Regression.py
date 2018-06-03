# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:18:56 2018

@author: bxl
"""
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 400), dim=1)
# y = x.pow(2) + x.pow(3) + 0.1*torch.rand(x.size())
y = x.pow(2) + x.pow(3)

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.relu((self.hidden(x)))
        x = self.predict(x)
        return x

net = Net(1, 5, 1)

plt.ion()
plt.show()

optimize = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.MSELoss()# 均方差

for t in range(400):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimize.zero_grad()
    loss.backward()
    optimize.step()
    
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20})
        plt.pause(0.5)
        
plt.ioff()
plt.show

