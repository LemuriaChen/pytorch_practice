
from __future__ import print_function
import torch
import numpy as np
import pandas as pd
from pprint import pprint

import torch.nn as nn
import torch.nn.functional as F


# Tensors初始化
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# x = torch.tensor([1, 2, 3])
# print(x)
#
# x = x.new_ones(1, 3, dtype=torch.long)
# print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5, 3)
print(x + y)


# Tensors操作符
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)

torch.add(x, y, out=result)

y.add_(x)

print(x[:, 1])

print(x[1, :])

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

a.transpose()

pd.DataFrame(a.transpose())

a.reshape(4, 3)

pd.DataFrame(a.reshape(4, 3))


image = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [1, 1, 1]]])
print(image)
print(image.shape)
pprint(image)


image2 = np.array(
    [[[[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [1, 1, 1]]], [[[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [1, 1, 1]]]]
)
pprint(image2)
print(image2.shape)

pprint(image2.transpose())
print(image2.transpose().shape)


pd.DataFrame(image2)


image2.transpose()

image2.reshape((-1, 6))


device = torch.device("cuda")  # a CUDA device object
y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
x = x.to(device)  # or just use strings ``.to("cuda")``
z = x + y
print(z)
print(z.to("cpu", torch.double))


# AUTOGRAD
x = torch.ones(2, 2, requires_grad=True)
print(a)

y = x + 2
print(y)

z = y * y * 3
out = z.mean()

print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()

print(x.grad)

torch.ones(3, 4).cuda()

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

y.backward()

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, requires_grad=True)
b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float, requires_grad=True)
c = a.mm(b)

c.backward()

c.backward(torch.ones_like(c))

a.grad
b.grad


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())


input_ = torch.randn(1, 1, 32, 32)
out = net(input_)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input_)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

loss.grad_fn

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input_)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input_ = torch.randn(45, 16, 44, 44)
output = m(input_)

print(output.shape)











