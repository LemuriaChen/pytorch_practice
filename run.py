import torch
from torch.autograd import Variable

print(torch.cuda.is_available())


gpu_info = Variable(torch.randn(3, 3)).cuda()
print(gpu_info)

cpu_info = gpu_info.cpu()
print(cpu_info)

