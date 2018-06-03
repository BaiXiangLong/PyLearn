# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_data_set = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_data_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print("Epoch:", epoch, "| Step:", step, "| batch_x:", batch_x.numpy(), "| batch_y:", batch_y.numpy())

