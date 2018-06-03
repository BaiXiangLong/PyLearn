# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

torch_data_set = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_data_set,
    batch_size=BATCH_SIZE,
    shuffle=2,
)



# optimizer = torch.optim.SGD()

