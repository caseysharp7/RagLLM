# experimental creation of my own model as training my model on purely CUDA could make the retrieval
# system significantly simpler, opted against as no means of training on large amounts of data
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
from torch.nn import Linear as lin
import torch.nn.functional as func

dev = torch.device("cuda")

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc1 = lin(1000, 2048)
        self.fc2 = lin(2048, 1024)
        self.fc3 = lin(1024, 512)
        self.fc4 = lin(512, 256)
        self.fc5 = lin(256, 128)
        self.fc6 = lin(128, 64)
        self.fc7 = lin(64, 32)
        self.fc8 = lin(32, 1)
    
    def forward(self, i):
        x = func.leaky_relu(self.fc1(i), negative_slope=0.05)
        x = func.leaky_relu(self.fc2(x), negative_slope=0.05)
        x = func.leaky_relu(self.fc3(x), negative_slope=0.05)
        x = func.leaky_relu(self.fc4(x), negative_slope=0.05)
        x = func.leaky_relu(self.fc5(x), negative_slope=0.05)
        x = func.leaky_relu(self.fc6(x), negative_slope=0.05)
        x = func.leaky_relu(self.fc7(x), negative_slope=0.05)
        x = self.fc8(x)
        return x

model = network().to(dev)

loss_func = nn.L1Loss()
optim = opt.Adam(model.parameters(), lr = 0.0003)
inputs = torch.randn(32, 1000).to(dev)
targets = torch.randn(32, 1).to(dev)
optim.zero_grad()

outputs = model(inputs)
loss = loss_func(outputs, targets)

loss.backward()
optim.step()
