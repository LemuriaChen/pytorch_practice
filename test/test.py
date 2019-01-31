
from __future__ import print_function
import torch
import numpy as np
import pandas as pd


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

print(image.shape)

image.reshape((-1, 6))

