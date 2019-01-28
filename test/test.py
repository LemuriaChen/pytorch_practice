
from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

print()

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([1, 2, 3])
print(x)

x = x.new_ones(1, 3, dtype=torch.long)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5, 3)
print(x + y)
