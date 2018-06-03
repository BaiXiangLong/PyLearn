# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  #(0, 1) (0-255)
    download=DOWNLOAD_MNIST,
)

# plot one example
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# for i in range(100):
#     plt.imshow(train_data.train_data[i].numpy())
#     plt.title("%i" % train_data.train_labels[i])
#     plt.show()

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False,)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[: 2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(      #(1, 28, 28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),              # 卷积层 ->(16, 28, 28)
            nn.ReLU(),                # ->(16, 28, 28)
            nn.MaxPool2d(kernel_size=2),  # pooling ->(16, 14, 14)
        )
        self.conv2 = nn.Sequential(# ->(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # ->(32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2) # ->(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        #print(x.size())
        # plt.imshow(x.data.numpy()[0][0])
        # plt.show()
        x = self.conv1(x)
        # plt.imshow(x.data.numpy()[0][0])
        # plt.show()
        x = self.conv2(x)           # (batch, 32, 7, 7)
        # plt.imshow(x.data.numpy()[0][0])
        # plt.show()
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn.forward(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn.forward(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print("Epoch: ", epoch, "| train loss: %.4f" % loss.data[0], "| test accuracy : %.4f" % accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

